import argparse
import os
import torch.distributed as dist
import torch
from exp.exp_stock import Exp_stock
from exp.exp_LLM4TS import Exp_Main
from exp.exp_ad import Exp_Anomaly_Detection

import random
import numpy as np
import pandas as pd
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '10010'

parser = argparse.ArgumentParser(description='Time Series Forecasting')

parser.add_argument('--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/mnt/petrelfs/chengdawei/lustre/wavlet/checkpoints/', help='location of model checkpoints')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--close_col', type=int, default=0, help='close column number')
parser.add_argument('--prev_close_col', type=int, default=4, help='prev_close column number')
parser.add_argument('--OT_col', type=int, default=6, help='OT column number')

parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# aLLMTS 
parser.add_argument('--is_llm', type=int, default=0, help='whether to use llm')
parser.add_argument('--pretrain', type=int, default=1, help='whether to use pretrained llm')
parser.add_argument('--freeze', type=int, default=1, help='whether to freeze specific part of the llm')
parser.add_argument('--llm_layers', type=int, default=1, help='the number of llm layers we use')
parser.add_argument('--mask_pt', type=int, default=0, help='mask pretrain ratio')
parser.add_argument('--llm', type=str, default='/mnt/petrelfs/chengdawei/lustre/LLaMA-Factory/gpt2', help='the llm checkpoint')
parser.add_argument('--attn_dropout', type=float, default=0, help='')
parser.add_argument('--proj_dropout', type=float, default=0, help='')
parser.add_argument('--res_attention', action='store_true', default=False, help='')

# sft
parser.add_argument('--sft', type=int, default=0, help='whether sft')
parser.add_argument('--sft_layers', type=str, default='null', help='the layers in llm needed to be trained')
parser.add_argument('--history_len', type=int, default=0, help='look-back window length')
parser.add_argument('--fft', type=int, default=0, help='fft')
parser.add_argument('--rand_init', type=int, default=0, help='rand_init')
# pt
parser.add_argument('--c_pt', type=int, default=0, help='whether continue pretrain')
parser.add_argument('--pt_layers', type=str, default='null', help='the layers in llm needed to be trained')
parser.add_argument('--pt_data', type=str, default='null', help='the dataset used in pretrain, use _ to separate')
parser.add_argument('--pt_sft', type=int, default=0, help='whether continue pretrain')
parser.add_argument('--pt_sft_base_dir', type=str, default='null', help='the base model dir for pt_sft')
parser.add_argument('--pt_sft_model', type=str, default='null', help='the base model for pt_sft')

# forecasting task
parser.add_argument('--seq_len', type=int, default=15, help='input sequence length')
parser.add_argument('--label_len', type=int, default=7, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--flag', type=str, default='train', help='Flag for data type: train, val, test, or pred')
parser.add_argument('--select_num', type=int, default=30, help='select_num')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=4, help='stride')
parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--notrans', action='store_true', default=False, help='stop using transformer')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', type=bool, default=True, help='whether to output attention in ecoder')#action='store_true'
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


#segment
parser.add_argument('--dynamic_segment', type=int, default=0, help='dynamic segment')
parser.add_argument('--segment_csv_path', type=str, default='/mnt/petrelfs/chengdawei/lustre/wavlet/segments/sisc_hs300_k12_l8-16_dba_kmpp_segmentation.csv', help='segment_csv_path')

#dwt
parser.add_argument('--wavelet', type=int, default=0, help='use wavelet transform')
parser.add_argument('--wavename', type=str, default='ldwt', help='use dwt or swt or ldwt')


if __name__ == '__main__':

    args = parser.parse_args()

    # random seed
    # fix_seed = args.random_seed
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # torch.cuda.manual_seed_all(fix_seed)
    # np.random.seed(fix_seed)
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
               
    #data process
    def filter_extreme_3sigma(series, n=3):
        mean = series.mean()
        std = series.std()
        max_range = mean + n * std
        min_range = mean - n * std
        return np.clip(series, min_range, max_range)

    def standardize_zscore(series):
        std = series.std()
        mean = series.mean()
        return (series - mean) / std

    def process_daily_df_std(df, feature_cols):
        df = df.copy()
        for c in feature_cols:
            df[c] = df[c].replace([np.inf, -np.inf], 0)
            df[c] = filter_extreme_3sigma(df[c])
            df[c] = standardize_zscore(df[c])
        return df
    
    def process_features(df_features, feature_cols):
        df_features_grouped = df_features.groupby('dt')
        res = []
        for dt in df_features_grouped.groups:
            df = df_features_grouped.get_group(dt)
            processed_df = process_daily_df_std(df, feature_cols)
            res.append(processed_df)
        df_features = pd.concat(res)
        df_features = df_features.dropna(subset=feature_cols)
        return df_features
    
    def process_group(group):
        group = group.sort_values(by='dt')
        group['OT'] = (group['close'] - group['prev_close']) / group['prev_close']
        return group

        
    data_name = args.data
    feature_cols = ['close','open','high','low','prev_close','volume']
    full_cols =   ['close','open','high','low','prev_close','volume','OT']
    args.close_col = full_cols.index('close')
    args.prev_close_col = full_cols.index('prev_close')
    args.OT_col = full_cols.index('OT')

    test_dt = pd.read_csv(f'/mnt/petrelfs/chengdawei/lustre/wavlet/dataset/{data_name}_dt.csv')
    df = pd.read_csv(f'/mnt/petrelfs/chengdawei/lustre/wavlet/dataset/{data_name}_org.csv')

    grouped = df.groupby('kdcode')
    processed_df = grouped.apply(process_group)
    df = processed_df.dropna().reset_index(drop=True)

    dataset = process_features(df,feature_cols)
    stocks = dataset['kdcode'].unique()
    dt_len = len(test_dt)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Main
    elif args.task_name == 'stock_forecast':
        Exp = Exp_stock
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection

    if args.is_training:
        for ii in range(args.itr):
            select_num = args.select_num
            output = []                
            setting = '{}_sl{}_pl{}_llml{}_lr{}_bs{}_percent{}_{}_{}'.format(
                args.model_id,
                args.seq_len,
                args.pred_len,
                args.llm_layers,
                args.learning_rate,
                args.batch_size,
                args.percent,
                args.des,ii)

            exp = Exp(args) 
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.use_multi_gpu and args.use_gpu and args.local_rank != 0:
                pass
            else:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                preds, trues = exp.test(setting)
                preds = preds[-dt_len:, :]
                trues = trues[-dt_len:, :]
                
                hold = np.empty((0, select_num))
                for i in range(trues.shape[0]):
                    # select = np.argsort(preds[i])[:select_num]
                    select = np.argsort(preds[i])[::-1][:select_num]
                    hold = np.vstack((hold, stocks[select]))
                hold_df = pd.DataFrame(hold)
                com_df = pd.concat([test_dt.reset_index(drop=True), hold_df], axis=1) 
                cols = ['dt']+[f'kdcode{i}' for i in range(1, select_num+1)] 
                com_df.columns = cols
                com_df.to_csv(f'/mnt/petrelfs/chengdawei/lustre/wavlet/{data_name}_data/hold_{data_name}_{ii}_{args.dynamic_segment}_{args.wavelet}_{args.llm_layers}_pt.csv', index=False)

                for index, row in com_df.iterrows():
                    dt = row['dt']
                    kdcode_columns = [f'kdcode{i}' for i in range(1, select_num+1)]
                    kd_codes = row[kdcode_columns].values            
                    ot_values = df[(df['kdcode'].isin(kd_codes))&(df['dt'] == dt)]['OT']
                    daily_return = ot_values.mean()
                    output.append([dt, daily_return])
            
                output_df = pd.DataFrame(output, columns=['datetime', 'daily_return'])
                output_df.to_csv(f'/mnt/petrelfs/chengdawei/lustre/wavlet/{data_name}_data/return_{data_name}_{ii}_{args.dynamic_segment}_{args.wavelet}_{args.llm_layers}_pt.csv', index=False)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_sl{}_pl{}_llml{}_lr{}_bs{}_percent{}_{}_{}'.format(
                args.model_id,
                args.seq_len,
                args.pred_len,
                args.llm_layers,
                args.learning_rate,
                args.batch_size,
                args.percent,
                args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
