from data_provider.data_loader import (Dataset_stock, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Weather, Dataset_Traffic, Dataset_Electricity, 
                                       Dataset_pretrain, PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, Dataset_M4, UEAloader)
from torch.utils.data import DataLoader
from data_provider.uea import collate_fn
import torch

data_dict = {'hs300': Dataset_stock, 'nd100': Dataset_stock, 'sp500': Dataset_stock, 'zz500': Dataset_stock, 
             'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
             'weather': Dataset_Weather, 'traffic': Dataset_Traffic, 'electricity': Dataset_Electricity, 'custom': Dataset_Custom,
             'pretrain': Dataset_pretrain, 'PSM': PSMSegLoader, 'MSL': MSLSegLoader, 'SMAP': SMAPSegLoader, 'SMD': SMDSegLoader,
             'SWaT': SWATSegLoader, 'm4': Dataset_M4, 'UEA': UEAloader}

def collate_fn(batch):
    seq_x = [item[0] for item in batch]  # (N, nstock, seq_len, nvars)
    seq_y = [item[1] for item in batch]  # (N, nstock, label_len + pred_len, nvars)
    seq_x_mark = [item[2] for item in batch]
    seq_y_mark = [item[3] for item in batch]

    seq_x = torch.stack(seq_x, dim=0) 
    seq_y = torch.stack(seq_y, dim=0)
    seq_x_mark = torch.stack(seq_x_mark, dim=0)  
    seq_y_mark = torch.stack(seq_y_mark, dim=0)

    B, N, L, M = seq_x.shape
    seq_x = seq_x.view(B*N, L, M)  # (N*nstock, seq_len, nvars)
    B, N, L, M = seq_y.shape 
    seq_y = seq_y.view(B*N, L, M)  # (N*nstock, label_len + pred_len, nvars)

    return seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag )
    elif args.task_name =='stock_forecast':
        shuffle_flag = False
        drop_last = False
        data_set = Data(
            configs=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=args.percent)  
    else:
        data_set = Data(
            configs=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=args.percent)  

    if args.use_multi_gpu and args.use_gpu and flag == 'train':
        if flag == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
            data_loader = DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
