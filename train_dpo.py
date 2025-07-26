# 文件名: train_dpo.py
import os
import json
import torch
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
# DPO 训练的核心组件从 TRL 库导入
from trl import DPOTrainer, DPOConfig

# --- 日志设置 (与您原脚本相同) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - [Rank %(process)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- 参数定义 ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. Options: 'auto', 'bfloat16', 'float16', 'float32'."})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the JSONL file for DPO training."})

# 关键改动：使用 DPOConfig 并添加 DPO 特有参数
# DPOConfig 继承自 TrainingArguments，因此我们可以直接使用它并添加我们需要的额外参数
@dataclass
class DPOTrainingArguments(DPOConfig):
    # DPOConfig 已经包含了所有 TrainingArguments 的字段
    # 我们可以在这里添加或覆盖字段
    # beta 是 DPO 的关键超参数，控制着对参考模型的偏离程度
    beta: float = field(default=0.1, metadata={"help": "The beta parameter for DPO loss."})
    # 您原有的参数可以直接通过 DPOConfig 进行设置，无需额外定义
    # 例如 --learning_rate, --num_train_epochs 等都在 DPOConfig 中

def setup_model_and_tokenizer(model_args: ModelArguments):
    """
    加载基础模型和分词器。
    对于DPO，我们只需要加载一个模型。参考模型将由DPOTrainer在内部自动创建。
    """
    logger.info(f"Loading model and tokenizer from {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # 对于使用 pad_token = eos_token 的模型，必须设置 padding_side='left'，以避免生成错误的文本
        tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype != "auto" else "auto",
        trust_remote_code=True
    )
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def process_dpo_data(data_args: DataArguments, tokenizer: AutoTokenizer):
    """
    重写数据处理函数以适配DPO格式。
    DPOTrainer 需要一个包含 'prompt', 'chosen', 'rejected' 三个列的数据集。
    我们不再手动进行 tokenization，DPOTrainer 会在内部完成。
    """
    logger.info(f"Processing DPO data from {data_args.data_path}...")
    
    # 使用 datasets 库直接加载 jsonl 文件，更高效
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    def format_prompt(example):
        # 你的 prompt 在 "prompt" 字段中，但它只是一个问题字符串。
        # 我们需要将其转换为模型期望的聊天格式。
        # 'chosen' 和 'rejected' 字段已经是完整的 assistant 回答，不需要处理。
        messages = [{"role": "user", "content": example["prompt"]}]
        # add_generation_prompt=True 会在末尾添加 'assistant'角色标识，提示模型开始生成
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt_str,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    original_columns = dataset.column_names
    dataset = dataset.map(
        format_prompt,
        remove_columns=original_columns, # 移除原始列，只保留 prompt, chosen, rejected
        desc="Formatting prompts for DPO"
    )
    
    logger.info(f"DPO data processed successfully. Number of examples: {len(dataset)}")
    logger.info(f"Example of a formatted sample:\n{dataset[0]}")
    return dataset

def main():
    # 关键改动：使用我们自定义的 DPOTrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"DPO Training parameters {training_args}")

    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 关键改动: DPO 不需要一个单独的引用模型副本。
    # DPOTrainer(ref_model=None) 会自动创建模型的副本作为引用模型，这是推荐的做法。
    # 这可以节省内存，并与 PEFT（如LoRA）更好地集成。
    ref_model = None

    if training_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    train_dataset = process_dpo_data(data_args, tokenizer)
    
    if not train_dataset:
        logger.error("Data processing returned an empty dataset. Exiting.")
        return

    # 关键改动：初始化 DPOTrainer
    # ... 在 main() 函数中
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("DPOTrainer initialized. Starting training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    

    trainer.save_model() 
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    logger.info(f"DPO Training finished. Final model and state saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()