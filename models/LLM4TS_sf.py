import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from layers.RevIN import RevIN

class Model(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        self.is_llm = configs.is_llm
        self.pretrain = configs.pretrain
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1
        self.d_ff = configs.d_ff

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
        self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.ln = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(
            configs.d_ff, 
            configs.enc_in, 
            bias=True)
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_len, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)


        if configs.freeze:
            if configs.c_pt:
                layers_train = configs.pt_layers.split('_')
            elif configs.sft:
                layers_train = configs.sft_layers.split('_')
            else:
                raise NotImplementedError

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
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = self.out_layer(dec_out)
        
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        
        return dec_out
