#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file "accelerate_config_lora.yaml" train_lora.py \
    --model_name_or_path "./models/Qwen2.5-7B-Instruct" \
    --data_path "./sft_data.jsonl" \
    --output_dir "./models/Qwen2.5-7B-Instruct-sft" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --bf16 True