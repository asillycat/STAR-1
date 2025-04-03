
# export WANDB_MODE=disabled

# bash run_sft.sh > output/all_10_1k_llama8b_distill_thinkflag1_03162150.out 2>&1

accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path star1_high.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1

# if distill model then base_flag=0 elif instruct model then base_flag=1
# if w/o think then think_flag=1