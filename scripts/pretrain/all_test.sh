export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs/pt_patch" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs/pt_patch
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs/pt_patch/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs/pt_patch/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness
fi

if [ ! -d "/mnt/petrelfs/chengdawei/lustre/aLLM4TS/checkpoints/pt_patch/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness" ]; then
    mkdir /mnt/petrelfs/chengdawei/lustre/aLLM4TS/checkpoints/pt_patch/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness
fi

model_name=LLM4TS_pt

root_path_name=/mnt/petrelfs/chengdawei/lustre/aLLM4TS/dataset/
data_path_name=null
data_name=pretrain
model_id=$data_name'_'$model_name
seq_len=1024
exp_des=basic
hf_model=gpt2

# 定义 GPU 数量
NUM_GPUS=8

# 使用 torchrun 启动
for seq_len in $seq_len
do
for label_len in 0 
do
for pred_len in $seq_len
do
for d_model in 768
do
for n_heads in 4
do
for e_layers in 4
do
for llm_layers in 6
do
for d_ff in 768
do
for lr in 0.0001 
do
for bs in 1024 
do
for percent in 100 
do
for pt_data in ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness 
do
for pt_layers in ln_wpe_attn_mlp
do
    exp_des=$pt_layers'_'$hf_model'_w_weight_s16'

    torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=10011 \
    --max_restarts=1 \
    /mnt/petrelfs/chengdawei/lustre/aLLM4TS/run_LLM4TS.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --llm /mnt/petrelfs/chengdawei/lustre/LLaMA-Factory/$hf_model \
    --affine 1 \
    --enc_in 1 \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 16 \
    --des $exp_des \
    --train_epochs 50 \
    --patience 5 \
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --c_pt 1 \
    --pt_data $pt_data \
    --pt_layers $pt_layers \
    --use_multi_gpu \
    --checkpoints /mnt/petrelfs/chengdawei/lustre/aLLM4TS/checkpoints/pt_patch/$pt_data \
    > /mnt/petrelfs/chengdawei/lustre/aLLM4TS/logs/pt_patch/test/$model_id'_sl'$seq_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1

done
done
done
done
done
done
done
done
done
done
done
done
done