
seq_len=31
model_name=LLM4TS_sft_zero 

root_path_name=/mnt/petrelfs/chengdawei/lustre/wavlet/dataset/
data_path_name=sp500_org.csv
data_name=sp500
model_id=$data_name'_'$model_name
wavename=ldwt
random_seed=2021

dynamic_segment=1
wavelet=1

for d_model in 768
do
for llm_layers in 12
do
for d_ff in 128
do
for bs in 1
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.0002
do
for sft_layers in ln_wpe
do
for pred_len in 1
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u /mnt/petrelfs/chengdawei/lustre/wavlet/run_LLM4TS.py \
    --task_name stock_forecast \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --pred_len $pred_len \
    --seq_len $seq_len \
    --revin 0 \
    --select_num 8 \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --llm /mnt/petrelfs/chengdawei/lustre/LLM/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 8\
    --stride 4\
    --des $exp_des \
    --train_epochs 10\
    --patience 10\
    --itr 10 \
    --batch_size $bs \
    --learning_rate $lr \
    --pt_sft 0 \
    --pt_sft_base_dir /mnt/petrelfs/chengdawei/lustre/wavlet/checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --dec_in 7 \
    --c_out 7 \
    --history_len 31 \
    --dynamic_segment $dynamic_segment \
    --wavelet $wavelet\
    --wavename $wavename\
    --segment_csv_path /mnt/petrelfs/chengdawei/lustre/wavlet/segments/sisc_$data_name'_k12_l8-16_dba_kmpp_segmentation.csv'\
    --checkpoints /mnt/petrelfs/chengdawei/lustre/wavlet/checkpoints/long-term-forecasting/$data_name \
    > /mnt/petrelfs/chengdawei/lustre/wavlet/logs/long-term-forecasting/$data_name/$model_id'_31_8_16_'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des'_dseg'$dynamic_segment'_wave'$wavelet'_tensor'.log 2>&1 
done
done
done
done
done
done
done
done
done