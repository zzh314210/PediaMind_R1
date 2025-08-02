#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Global Batch Size = per_device * num_gpus * grad_accum = 4 * 8 * 8 = 256
accelerate launch --config_file "accelerate_config_grpo.yaml" train_grpo.py \
    --model_name_or_path "./models/Qwen2.5-7B-Instruct-sft" \
    --data_path "./dataset/grpodata/grpodataset.jsonl" \
    --output_dir "./models/Qwen2.5-7B-Instruct-sft-grpo" \
    --run_name "qwen-baby-grpo" \
    --learning_rate 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --num_train_epochs 3 \
    --save_steps 20 \
    --max_grad_norm 0.5 \
    --report_to "tensorboard" \
    --temperature 1.0 \
    --beta 0.005 \