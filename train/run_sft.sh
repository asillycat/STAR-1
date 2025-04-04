
# export WANDB_MODE=disabled

accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --train_bsz_per_gpu 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ../data/STAR-1.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1

# if distill model then base_flag=0 elif instruct model then base_flag=1
# if w/o think then think_flag=0 else default=1
# train_bsz_per_gpu * num_processes should be 8 to keep the batchsize as 128
# you change the model_path to different model or change the data_path to use different finetune data
