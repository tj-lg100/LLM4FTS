# 单级小波
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from transformers import AutoModel, AutoTokenizer
from .modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
from layers.RevIN import RevIN

class Model(nn.Module):
    
    def __init__(self, configs, device):
        super().__init__()
        self.is_llm = configs.is_llm
        self.pretrain = configs.pretrain
        
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.history_len = configs.history_len
        self.patch_num = (configs.history_len - self.patch_len) // self.stride + 1 + (configs.pred_len - self.patch_len) // self.stride + 1 #130
        self.num_loss_patch = (configs.pred_len  - self.patch_len) // self.stride + 1  #89
        self.revin = configs.revin
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
        self.wave = 'db1'  # 选择小波基
        self.J = 1  # 分解级数
        self.device = device
        self.dwt = DWT1DForward(wave=self.wave, J=self.J).to(self.device)
        self.idwt = DWT1DInverse(wave=self.wave).to(self.device)

        if configs.is_llm:
            if configs.pretrain:
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model.from_pretrained(configs.llm, output_attentions=True, output_hidden_states=True)  
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel.from_pretrained(configs.llm, output_attentions=True, output_hidden_states=True)  
                else:
                    raise NotImplementedError
            else:
                print("------------------no pretrain------------------")
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model(GPT2Config())
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel(LlamaConfig())
                else:
                    raise NotImplementedError
            if "gpt2" in configs.llm:
                self.gpt.h = self.gpt.h[:configs.llm_layers]
                print("gpt2 = {}".format(self.gpt))
            elif "llama" in configs.llm:
                self.gpt.layers = self.gpt.layers[:configs.llm_layers]
                print("llama2 = {}".format(self.gpt))
            else:
                raise NotImplementedError
        
        self.in_layer = nn.Linear(self.patch_len, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, self.patch_len)


        if configs.freeze and configs.pretrain:
            if configs.c_pt:
                layers_train = configs.pt_layers.split('_')
            elif configs.sft:
                layers_train = configs.sft_layers.split('_')
            else:
                layers_train = '__'

            for i, (name, param) in enumerate(self.gpt.named_parameters()):
                tag = 0
                for layer_train in layers_train:
                    if layer_train in name:
                        tag = 1
                if tag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        for layer in (self.gpt, self.in_layer, self.out_layer, self.revin_layer):
            layer.train()
        print(f"configs.history_len: {configs.history_len}")
        print(f"configs.pred_len: {configs.pred_len}")
        print(f"self.patch_len: {self.patch_len}")
        print(f"self.stride: {self.stride}")
        print(f"self.patch_num (before calculation): {self.patch_num}")

    def forward(self, x, y):
        B, L, M = x.shape
        # print(f"x.shape:{x.shape}")
        # print(f"y.shape:{y.shape}")

        if self.revin: 
            x = self.revin_layer(x, 'norm')
            y = self.revin_layer._normalize(y)
        
        x = rearrange(x, 'b l m -> b m l')
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x = rearrange(x, 'b m n p -> (b m) n p')

        y = rearrange(y, 'b l m -> b m l')
        # y = y.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # y = rearrange(y, 'b m n p -> (b m) n p')

        # 对 x 和 y 进行小波变换
        cA_x, cD_x = self.dwt(x)  # cA_x 和 cD_x 为 (B, M, L//2)
        cA_y, cD_y = self.dwt(y)
        cD_x = cD_x[0]
        cD_y = cD_y[0]
        

        cA_x = cA_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        cA_x = rearrange(cA_x, 'b m n p -> (b m) n p')
        mean_cA_x = cA_x.mean(dim=1, keepdim=True) #(16*7,1,16)
        #print(f"cA_x.shape:{cA_x.shape}") #(16*7,20,16)

        cA_y = cA_y.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        cA_y = rearrange(cA_y, 'b m n p -> (b m) n p')
        #mean_cA_y = cA_y.mean(dim=1, keepdim=True)
        #print(f"cA_y.shape:{cA_y.shape}") #(16*7,44,16)

        cD_x = cD_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        cD_x = rearrange(cD_x, 'b m n p -> (b m) n p')
        mean_cD_x = cD_x.mean(dim=1, keepdim=True) #(16*7,1,16)
        #print(f"cD_x.shape:{cD_x.shape}")

        cD_y = cD_y.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        cD_y = rearrange(cD_y, 'b m n p -> (b m) n p')
        #mean_cD_y = cD_y.mean(dim=1, keepdim=True)
        #print(f"cD_y.shape:{cD_y.shape}")

        ts_inputs1 = torch.cat([self.in_layer(cA_x), self.in_layer(cA_y)], dim=-2) #20+44
        ts_inputs2 = torch.cat([self.in_layer(cD_x), self.in_layer(cD_y)], dim=-2) #20+44
        self.num_loss_patch=(self.num_loss_patch//2)

        ##
        my_attn_mask1 = torch.ones((B*M, ts_inputs1.shape[-2], ts_inputs1.shape[-2])).to(ts_inputs1.device).bool()
        my_attn_mask1[:, :, -self.num_loss_patch:] = 0
        for i in range(-self.num_loss_patch, 0):
            my_attn_mask1[:, i, i] = 1

        outputs1 = self.gpt(inputs_embeds=ts_inputs1, attention_mask=my_attn_mask1).last_hidden_state
        outputs1 = self.out_layer(outputs1) #(16*7,64,16)
        outputs1 = torch.cat((mean_cA_x, outputs1), dim=1) #(16*7,65,16)
        outputs1 = rearrange(outputs1, '(b m) n p -> b m (n p)', b=B, m=M) #(16,7,65*16)
        #print(f"outputs1:{outputs1.shape}")   #(16,7,1040)

        ##
        my_attn_mask2 = torch.ones((B*M, ts_inputs2.shape[-2], ts_inputs2.shape[-2])).to(ts_inputs2.device).bool()
        my_attn_mask2[:, :, -self.num_loss_patch:] = 0
        for i in range(-self.num_loss_patch, 0):
            my_attn_mask2[:, i, i] = 1

        outputs2 = self.gpt(inputs_embeds=ts_inputs2, attention_mask=my_attn_mask2).last_hidden_state
        outputs2 = self.out_layer(outputs2) #(16*7,64,16) 
        outputs2 = torch.cat((mean_cD_x, outputs2), dim=1)
        outputs2 = rearrange(outputs2, '(b m) n p -> b m (n p)', b=B, m=M)  #(16,7,65*16)
        #print(f"outputs2:{outputs2.shape}") 

        #进行逆小波变换
        outputs = self.idwt((outputs1, [outputs2]))  # cD 需要以列表形式提供
        #print(f"小波变换后的形状:{outputs.shape}")

        outputs = rearrange(outputs, 'b m (n p) -> b (n p) m', b=B, m=M, n=self.patch_num, p=self.patch_len)
        if self.revin: 
            outputs = self.revin_layer(outputs, 'denorm')
        outputs = rearrange(outputs, 'b (n p) m -> b m n p', n=self.patch_num ,p=self.patch_len)

        return outputs
    