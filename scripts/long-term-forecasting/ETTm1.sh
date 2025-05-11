export CUDA_VISIBLE_DEVICES=0,1,2

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/wavelettest/logs" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/wavelettest/logs
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/wavelettest/logs/long-term-forecasting" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/wavelettest/logs/long-term-forecasting
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/long-term-forecasting" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/long-term-forecasting
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/wavelettest/logs/long-term-forecasting/ETTm1" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/wavelettest/logs/long-term-forecasting/ETTm1
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/long-term-forecasting/ETTm1" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/long-term-forecasting/ETTm1
fi



model_name=LLM4TS_sft_zero

root_path_name=/mnt/petrelfs/chengdawei/lustre/wavelettest/dataset/
data_path_name=ETTm1.csv
data_name=ETTm1
model_id=$data_name'_'$model_name

for pred_len in 720
do
for d_model in 768
do
for llm_layers in 6
do
for d_ff in 768
do
for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.005
do
for sft_layers in ln_wpe 
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u /mnt/petrelfs/chengdawei/lustre/wavelettest/run_LLM4TS.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --pred_len $pred_len \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --llm /mnt/petrelfs/chengdawei/lustre/LLaMA-Factory/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des $exp_des \
    --train_epochs 30\
    --patience 5\
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --pt_sft 1 \
    --pt_sft_base_dir /mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --history_len 336 \
    --checkpoints /mnt/petrelfs/chengdawei/lustre/wavelettest/checkpoints/long-term-forecasting/ETTm1 \
    > /mnt/petrelfs/chengdawei/lustre/wavelettest/logs/long-term-forecasting/ETTm1/$model_id'_336_'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done
done
done
done
done