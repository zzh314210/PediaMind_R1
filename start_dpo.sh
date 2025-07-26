#!/bin/bash
# 文件名: start_dpo.sh

accelerate launch --config_file "accelerate_config.yaml" train_dpo.py \
    --model_name_or_path "./models/Qwen2.5-7B-Instruct-sft-sft" \
    --data_path "./dataset/dpodata/dpo_train_data.jsonl" \
    --output_dir "./models/Qwen2.5-7B-Instruct-sft-sft-dpo" \
    --max_length 1024 \
    --max_prompt_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --log_level "info" \
    --bf16 True \
    --max_grad_norm 1.0 \
    --beta 0.1