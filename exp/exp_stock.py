from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LLM4TS_pt, LLM4TS_sft_zero
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from scipy.stats import spearmanr, rankdata

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from einops import rearrange
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

from thop import profile
import time
import math

warnings.filterwarnings('ignore')

class Exp_stock(Exp_Basic):
    def __init__(self, args):
        super(Exp_stock, self).__init__(args)
        self.patch_len = self.args.patch_len
        self.stride = self.args.stride
        self.patch_num = int((self.args.seq_len - self.patch_len)/self.stride + 1)
        
    def _build_model(self):
        model_dict = {
            'LLM4TS_pt': LLM4TS_pt,
            'LLM4TS_sft_zero': LLM4TS_sft_zero
        }
        model = model_dict[self.args.model].Model(self.args).float()  # device=self.device

        if self.args.pt_sft:
            if self.args.use_multi_gpu and self.args.use_gpu:
                pt_model = torch.load(os.path.join(self.args.pt_sft_base_dir, self.args.pt_sft_model + "/checkpoint.pth"), map_location='cpu')
            else:
                pt_model = torch.load(os.path.join(self.args.pt_sft_base_dir, self.args.pt_sft_model + "/checkpoint.pth"))
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in pt_model.items() if k in model_dict.keys()}

            if self.args.model == "LLM4TS_sft_zero":
                state_dict.pop('revin_layer.affine_weight', None)
                state_dict.pop('revin_layer.affine_bias', None)
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
            else:
                state_dict.pop('revin_layer.affine_weight', None)
                state_dict.pop('revin_layer.affine_bias', None)
                state_dict.pop('out_layer.weight', None)
                state_dict.pop('out_layer.bias', None)
                
            del pt_model
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            criterion = nn.MSELoss()
            return criterion

    
    def vali(self, vali_data, vali_loader): # criterion

        MSE_loss = nn.MSELoss()
        def pairwise_loss(preds, labels, margin=0.1):
            preds = preds.view(-1)
            labels = labels.view(-1)
            assert preds.size() == labels.size()
            diff_preds = preds[None,:] - preds[:, None]
            diff_labels = labels[None, :] - labels[:, None]
            
            mask = (diff_preds * diff_labels < 0).type(torch.float32)
            hinge_loss = torch.nn.functional.relu(margin - diff_preds).pow(2)
            loss = mask * hinge_loss
            return loss.sum()
        
        def criterloss(preds, labels, alpha=1, beta=2e-5):
            return alpha*MSE_loss(preds, labels)   # + beta*pairwise_loss(preds, labels)
        
        total_loss = 0
        preds = []
        trues = []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                time_now = time.time()
 
                if "LLM4TS_sft_zero" == self.args.model:
                    batch_x_ = batch_x[:, -batch_y.shape[-2]:, :] 
                    if self.args.rand_init:
                        batch_x_ = torch.randn_like(batch_y).to(self.device, dtype=torch.float)
                    if self.args.fft:
                        batch_x_ = batch_x_.permute(0, 2, 1).cpu().numpy()
                        fft_result = fft(batch_x_)
                        num_components = self.args.fft
                        indices = np.argsort(np.abs(fft_result))[::-1][:, :, -num_components:]
                        fft_result_filtered = np.zeros_like(fft_result)
                        fft_result_filtered[np.arange(fft_result_filtered.shape[0])[:, None, None], 
                                            np.arange(fft_result_filtered.shape[1])[:, None], indices] = fft_result[
                                                np.arange(fft_result_filtered.shape[0])[:, None, None], 
                                                np.arange(fft_result_filtered.shape[1])[:, None], indices]
                        reconstructed_signal = ifft(fft_result_filtered).real
                        batch_x_ = torch.tensor(reconstructed_signal).to(self.device, dtype=torch.float).permute(0, 2, 1)
                    
                    batch_x = batch_x[:, -self.args.history_len:, :]
                    outputs = self.model(batch_x, batch_x_,i)  # (bs*nstocks, nvars, patch_num=3, patch_len)
                else:
                    if self.args.mask_pt > 0:
                        B, T, N = batch_x.shape
                        mask_rate = self.args.mask_pt / 100.0
                        patch_num = (T - self.args.patch_len) // self.args.stride + 1
                        mask = torch.rand((B, N, patch_num,)).to(self.device)
                        mask[mask <= mask_rate] = 0 
                        mask[mask > mask_rate] = 1  
                        outputs = self.model(batch_x, mask)  # (bs*nstocks, nvars, patch_num=3, patch_len)
                    else:
                        outputs = self.model(batch_x)
                        
                total_time += time.time() - time_now
                
                if "LLM4TS_sft_zero" == self.args.model:
                    num_loss_patch = (self.args.label_len+self.args.pred_len-self.args.patch_len)//self.args.stride + 1
                    outputs = outputs[:, :, -num_loss_patch:, :]
                    outputs = rearrange(outputs, 'b m n p -> b (n p) m', n=num_loss_patch ,p=self.patch_len)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # close1 = outputs[:, :, self.args.close_col]
                # prev_close1 = outputs[:, :, self.args.prev_close_col]
                # close2 = batch_y[:, :, self.args.close_col]
                # prev_close2 = batch_y[:, :, self.args.prev_close_col]
                # OT_value_pred = (close1 - prev_close1) / prev_close1  #(nstocks, 1)
                # OT_value_true = (close2 - prev_close2) / prev_close2  #(nstocks, 1)
                OT_value_pred = outputs[:, :, self.args.OT_col]  #(nstocks, 1)
                OT_value_true = batch_y[:, :, self.args.OT_col]  #(nstocks, 1)
                
                pred = OT_value_pred.detach().cpu()  #(nstocks, 1)
                true = OT_value_true.detach().cpu()  #(nstocks, 1)

                loss = criterloss(pred, true)
                total_loss += loss.item()
                preds.append(np.array(pred.squeeze().tolist()))
                trues.append(np.array(true.squeeze().tolist()))
        
        preds=np.array(preds)  #(N,nstocks)
        trues=np.array(trues)  #(N,nstocks)

        daily_rankICs = np.array([spearmanr(rankdata(preds[i]), 
                                            rankdata(trues[i])).correlation for i in range(preds.shape[0])])
        mean_rankIC = np.mean(daily_rankICs)
        total_loss = total_loss/len(vali_loader)
        print('vali time: ', total_time)
        self.model.train()
        return mean_rankIC, total_loss

    def train(self, setting):
        
        # MSE_loss = nn.MSELoss()
        # def pairwise_loss(preds, labels, margin=0.1):
        #     preds = preds.view(-1)
        #     labels = labels.view(-1)
        #     assert preds.size() == labels.size()
        #     diff_preds = preds[None,:] - preds[:, None]
        #     diff_labels = labels[None, :] - labels[:, None]
            
        #     mask = (diff_preds * diff_labels < 0).type(torch.float32)
        #     hinge_loss = torch.nn.functional.relu(margin - diff_preds).pow(2)
        #     loss = mask * hinge_loss
        #     return loss.sum()
        
        # def criterloss(preds, labels, alpha=1, beta=2e-5):
        #     return alpha*MSE_loss(preds, labels)   + beta*pairwise_loss(preds, labels)


        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        x, y, x_mark, y_mark  = next(iter(test_loader))
        x=x.to(self.device, dtype=torch.float)
        y=y.to(self.device, dtype=torch.float)
        # from torchinfo import summary
        # if "LLM4TS_sft_zero" == self.args.model:
        #     summary(self.model, input_size=[x[:, -self.args.history_len:, :].size(), y.size()])
        # else:
        #     summary(self.model, input_size=x.size())
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        low_weights_history = []  # 存储所有epoch的低通滤波器权重
        high_weights_history = [] # 存储所有epoch的高通滤波器权重
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        max_ic = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device, dtype=torch.float)
                batch_y = batch_y.to(self.device, dtype=torch.float)

                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if "LLM4TS_sft_zero" == self.args.model:
                        batch_x_ = batch_x[:, -batch_y.shape[-2]:, :]
                        if self.args.rand_init:
                            batch_x_ = torch.randn_like(batch_y).to(self.device, dtype=torch.float)
                        if self.args.fft:
                            batch_x_ = batch_x_.permute(0, 2, 1).cpu().numpy()
                            fft_result = fft(batch_x_)
                            num_components = self.args.fft
                            indices = np.argsort(np.abs(fft_result))[::-1][:, :, -num_components:]
                            fft_result_filtered = np.zeros_like(fft_result)
                            fft_result_filtered[np.arange(fft_result_filtered.shape[0])[:, None, None], np.arange(fft_result_filtered.shape[1])[:, None], 
                                                indices] = fft_result[np.arange(fft_result_filtered.shape[0])[:, None, None], 
                                                                      np.arange(fft_result_filtered.shape[1])[:, None], indices]
                            reconstructed_signal = ifft(fft_result_filtered).real
                            batch_x_ = torch.tensor(reconstructed_signal).to(self.device, dtype=torch.float).permute(0, 2, 1)
                        
                        batch_x = batch_x[:, -self.args.history_len:, :]
                        outputs = self.model(batch_x, batch_x_, i)      # (bs*nstocks, nvars, patch_num=3, patch_len)                                      
                    else:
                        if self.args.mask_pt > 0:
                            B, T, N = batch_x.shape
                            mask_rate = self.args.mask_pt / 100.0
                            patch_num = (T - self.args.patch_len) // self.args.stride + 1
                            mask = torch.rand((B, N, patch_num,)).to(self.device)
                            mask[mask <= mask_rate] = 0 
                            mask[mask > mask_rate] = 1  
                            outputs = self.model(batch_x, mask)      # (bs*nstocks, nvars, patch_num=3, patch_len)
                        else:
                            outputs = self.model(batch_x)            # (bs*nstocks, nvars, patch_num=3, patch_len)     

                    if "LLM4TS_sft_zero" == self.args.model:
                        num_loss_patch = int((self.args.label_len+self.args.pred_len-self.args.patch_len)/self.args.stride + 1)
                        outputs = outputs[:, :, -num_loss_patch:, :] # (bs*nstocks, nvars, patch_num=1, patch_len) 
                        
                        new_y = batch_y.permute(0, 2, 1)                           
                        new_y = new_y.unfold(dimension=-1, size=self.args.patch_len, step=self.args.stride)

                    if "LLM4TS_sft_zero" == self.args.model:
                        loss = 0
                        for kind in [ 1 ]:
                            loss += criterion(outputs[:, -1, :kind, -1], new_y[:, -1, :kind, -1])
                            loss_num_dict={'1': 1.}
                        loss = loss / loss_num_dict[str(self.args.pred_len)]
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            low_weights_history.append(self.model.ldwt.conv_low.weight.data.cpu().numpy())
            high_weights_history.append(self.model.ldwt.conv_high.weight.data.cpu().numpy())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    
            print('low_weights:',self.model.ldwt.conv_low.weight.data.cpu().numpy())
            print('high_weights:',self.model.ldwt.conv_high.weight.data.cpu().numpy())
            
            train_loss = np.average(train_loss)

            if self.args.use_multi_gpu and self.args.use_gpu and self.args.local_rank != 0:
                pass
            else:
                mean_rankIC, vali_loss = self.vali(vali_data, vali_loader)  # criterion  
                
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth') 
                # test_loss = self.vali(test_data, test_loader)  # criterion
                # early_stopping(mean_rankIC, self.model, path)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Rank IC: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, mean_rankIC))
               
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        np.save(f"/mnt/petrelfs/chengdawei/lustre/wavlet/conv_weights/low_weights_histor_{self.args.data}.npy", np.stack(low_weights_history))  # 形状 [num_epochs, 7, 1, 8]
        np.save(f"/mnt/petrelfs/chengdawei/lustre/wavlet/conv_weights/high_weights_history_{self.args.data}.npy", np.stack(high_weights_history))
        print("权重数据已保存至 low_weights_history.npy 和 high_weights_history.npy")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        path1 = '/mnt/petrelfs/chengdawei/lustre/wavlet/test_ckpt/'
        path2 = '/mnt/petrelfs/chengdawei/lustre/wavlet/test_results/'
        path3 = '/mnt/petrelfs/chengdawei/lustre/wavlet/results/'

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path1, setting[:-2], 'checkpoint.pth')))
        folder_path = path2 + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        preds = []
        trues = []
        inputx = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_y_mark = batch_y_mark.float().to(self.device)

                if "LLM4TS_sft_zero" == self.args.model:
                    batch_x_ = batch_x[:, -batch_y.shape[-2]:, :]
                    if self.args.rand_init:
                        batch_x_ = torch.randn_like(batch_y).to(self.device, dtype=torch.float)
                    if self.args.fft:
                        batch_x_ = batch_x_.permute(0, 2, 1).cpu().numpy()
                        fft_result = fft(batch_x_)
                        num_components = self.args.fft
                        indices = np.argsort(np.abs(fft_result))[::-1][:, :, -num_components:]
                        fft_result_filtered = np.zeros_like(fft_result)
                        fft_result_filtered[np.arange(fft_result_filtered.shape[0])[:, None, None], np.arange(fft_result_filtered.shape[1])[:, None], 
                                            indices] = fft_result[np.arange(fft_result_filtered.shape[0])[:, None, None], 
                                                                  np.arange(fft_result_filtered.shape[1])[:, None], indices]
                        reconstructed_signal = ifft(fft_result_filtered).real
                        batch_x_ = torch.tensor(reconstructed_signal).to(self.device, dtype=torch.float).permute(0, 2, 1)

                    batch_x = batch_x[:, -self.args.history_len:, :]
                    outputs = self.model(batch_x, batch_x_,i)           # (bs*nstocks, nvars, patch_num=3, patch_len)  
                else:
                    if self.args.mask_pt > 0:
                        B, T, N = batch_x.shape
                        mask_rate = self.args.mask_pt / 100.0
                        patch_num = (T - self.args.patch_len) // self.args.stride + 1
                        mask = torch.rand((B, N, patch_num,)).to(self.device)
                        mask[mask <= mask_rate] = 0 
                        mask[mask > mask_rate] = 1
                        outputs = self.model(batch_x, mask)      # (bs*nstocks, nvars, patch_num=3, patch_len)  
                    else:
                        outputs = self.model(batch_x)
                
                if "LLM4TS_pt" == self.args.model:
                    _, L, _ = batch_y.size()
                    B, C, n, p = outputs.size()
                    outputs = outputs.detach().cpu()
                    y_hat = torch.zeros(B, C, L)
                    for b in range(B):
                        for c in range(C):
                            last_end = 0
                            for i in range(n-1):
                                start = i * self.args.stride
                                end = start + self.args.patch_len
                                if last_end<=start:
                                    y_hat[b, c, start:end] += outputs[b, c, i, :]
                                else:
                                    y_hat[b, c, start:last_end] = (y_hat[b, c, start:last_end] + outputs[b, c, i, :(last_end - start)]) / 2
                                    y_hat[b, c, last_end:end] += outputs[b, c, i, (last_end - start):]
                                last_end = end
                    y_hat = y_hat.permute(0, 2, 1)        # (bs*nstocks, label_len+pred_len, nvars)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = y_hat[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                elif "LLM4TS_sft_zero" == self.args.model:
                    num_loss_patch = (self.args.label_len+self.args.pred_len-self.args.patch_len)//self.args.stride + 1
                    outputs = outputs[:, :, -num_loss_patch:, :]   # (bs*nstocks, nvars, patch_num=1, patch_len)

                    B, C, n, p = outputs.size()
                    _, L, _ = batch_y.size()
                    outputs = outputs.detach().cpu()
                    y_hat = torch.zeros(B, C, L)
                    last_end = 0
                    for j in range(n):
                        start = j * self.args.stride
                        end = start + self.args.patch_len
                        if last_end<=start:
                            y_hat[:, :, start:end] += outputs[:, :, j, :]
                        else:
                            y_hat[:, :, start:last_end] = (y_hat[:, :, start:last_end] + outputs[:, :, j, :(last_end - start)]) / 2
                            y_hat[:, :, last_end:end] += outputs[:, :, j, (last_end - start):]
                        last_end = end
                    y_hat = y_hat.permute(0, 2, 1)       # (bs*nstocks, label_len + pred_len, nvars)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = y_hat[:, -self.args.pred_len:, f_dim:]                    # (bs*nstocks, pred_len, nvars)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # (bs*nstocks, pred_len, nvars)

                    # close1 = outputs[:, :, self.args.close_col]
                    # prev_close1 = outputs[:, :, self.args.prev_close_col]
                    # close2 = batch_y[:, :, self.args.close_col]
                    # prev_close2 = batch_y[:, :, self.args.prev_close_col]
                    # OT_value_pred = (close1 - prev_close1) / prev_close1  #(nstocks, 1)
                    # OT_value_true = (close2 - prev_close2) / prev_close2  #(nstocks, 1)
                    OT_value_pred = outputs[:, :, self.args.OT_col]  #(nstocks, 1)
                    OT_value_true = batch_y[:, :, self.args.OT_col]  #(nstocks, 1)

                    pred = OT_value_pred.detach().cpu()
                    true = OT_value_true.detach().cpu()
                    preds.append(np.array(pred.squeeze().tolist()))
                    trues.append(np.array(true.squeeze().tolist()))

                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 5 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))
                
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.array(preds)  # (N,nstocks)
        trues = np.array(trues)  # (N,nstocks)
        daily_rankICs = np.array([spearmanr(rankdata(preds[i]), 
                                            rankdata(trues[i])).correlation for i in range(preds.shape[0])])
        mean_rankIC = np.mean(daily_rankICs)
        std_dev_rankIC = np.std(daily_rankICs)
        RankICIR = mean_rankIC / std_dev_rankIC

        inputx = np.array(inputx)
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        folder_path = path3 + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f"begin calculating...")
        mae, mse, rmse, mape, mspe, rse = metric(preds, trues)
        mae = mae/len(test_loader)
        mse = mse/len(test_loader) 
        rse = rse/len(test_loader)
        print('mse:{}, mae:{}, rse:{}, Rank IC:{}, ICIR:{}'.format(mse, mae, rse, mean_rankIC, RankICIR))
        
        file_path = os.path.join(path3, 'result.txt')
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")  
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse)) 
            f.write('\n')  
            f.write('\n') 
        np.save(os.path.join(folder_path, 'pred.npy'), preds)

        return preds, trues
