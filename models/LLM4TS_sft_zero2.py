import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from .modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
from layers.RevIN import RevIN
from layers.PairNorm import PairNorm
class Model(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        self.is_llm = configs.is_llm
        self.pretrain = configs.pretrain
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.history_len = configs.history_len
        self.patch_num = int((configs.history_len-self.patch_len)/self.stride+1) + int((configs.label_len+configs.pred_len-self.patch_len)/self.stride+1)
        self.num_loss_patch = int((configs.label_len + configs.pred_len - self.patch_len)/self.stride + 1)
        
        self.n_vars = configs.enc_in
        self.head_nf = configs.d_model
        self.segment_positions = None
        
        self.dynamic_segment =  configs.dynamic_segment

        if configs.dynamic_segment:
            self.segment_positions = pd.read_csv(configs.segment_csv_path, header=None, index_col=0).values.flatten()

        self.revin = configs.revin
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)

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
        self.pn = PairNorm(mode='PN-SI')

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
                    
        for layer in (self.gpt, self.in_layer, self.out_layer,self.pn):  # self.revin_layer
            layer.train()
        

    def forward(self, x, y, batch_index):
        B, L, M = x.shape
        if self.revin: 
            x = self.revin_layer(x, 'norm')
            y = self.revin_layer._normalize(y)
        
        y = rearrange(y, 'b l m -> b m l')
        y = y.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (bs*nstock, nvars, 1, patch_len)
        y = rearrange(y, 'b m n p -> (b m) n p')


        # do patching
        if self.dynamic_segment:
            start_index = batch_index * L  # 当前 batch 的起始位置
            end_index = start_index + L   # 当前 batch 的结束位置
            csv_segment_starts = [pos - start_index for pos in self.segment_positions if start_index <= pos < end_index]

            stride_starts = list(range(0, L - self.patch_len + 1, self.stride))

            segment_starts = sorted(set(csv_segment_starts + stride_starts))

            x_patches = []
            for start in segment_starts:
                if start + self.patch_len > L:
                    continue
                x_patches.append(x[:, start:start + self.patch_len, :])
                
            if not x_patches:
                raise ValueError(f"No valid segments found for batch index {batch_index} in range [{start_index}, {end_index}]")
            
            x = torch.stack(x_patches, dim=1)  # (B, num_patches, patch_len, M)
            
            x = rearrange(x, 'b np l m -> (b m) np l')  # (B*M, num_patches, patch_len)

            ts_inputs = torch.cat([self.in_layer(x), self.in_layer(y)], dim=-2)  # (B*M, num_patches, d_model)

        else:
            x = rearrange(x, 'b l m -> b m l')
            
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (bs*nstock, nvars, 2, patch_len)
            x = rearrange(x, 'b m n p -> (b m) n p')
           
            ts_inputs = torch.cat([self.in_layer(x),self.in_layer(y)], dim=-2) # (bs*nstock*nvars, 3, d_model)


        
        my_attn_mask = torch.ones((B*M, ts_inputs.shape[-2], ts_inputs.shape[-2])).to(ts_inputs.device).bool()
        my_attn_mask[:, :, -self.num_loss_patch:] = 0
        for i in range(-self.num_loss_patch, 0):
            my_attn_mask[:, i, i] = 1

        outputs = self.gpt(inputs_embeds=ts_inputs, attention_mask=my_attn_mask).last_hidden_state # (bs*nstock*nvars, 3, d_model)
        outputs = self.pn(outputs)
        outputs = self.out_layer(outputs)                                 # (bs*nstock*nvars, 3, patch_len)
        outputs = rearrange(outputs, '(b m) n p -> b (n p) m', b=B, m=M)  
        if self.revin: 
            outputs = self.revin_layer(outputs, 'denorm')
        outputs = rearrange(outputs, 'b (n p) m -> b m n p',p=self.patch_len)   # (bs*nstock, nvars, 3, patch_len)
        
        return outputs
