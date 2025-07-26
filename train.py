import os
import json
import torch
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - [Rank %(process)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. Options: 'auto', 'bfloat16', 'float16', 'float32'."})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the JSONL file for training."})
    max_seq_length: int = field(default=1024, metadata={"help": "The maximum total input sequence length after tokenization."})

def setup_model_and_tokenizer(model_args: ModelArguments):
    logger.info(f"Loading model and tokenizer from {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype != "auto" else "auto",
        trust_remote_code=True
    )
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def process_data(data_args: DataArguments, tokenizer: AutoTokenizer):
    logger.info(f"Processing data from {data_args.data_path}...")
    with open(data_args.data_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    def format_to_chat_prompt(example):
        messages = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["response"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": prompt}

    formatted_data_list = [format_to_chat_prompt(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data_list)

    def tokenize_function(examples):
        # 修改点：将 padding=False 改为 padding="max_length"
        # 这会确保所有序列在预处理阶段就被填充到相同的长度，解决了在批处理时因长度不一而导致的错误。
        tokenized_output = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=data_args.max_seq_length,
            padding="max_length" 
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )
    logger.info(f"Data tokenized successfully. Number of examples: {len(tokenized_dataset)}")
    return tokenized_dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training/evaluation parameters {training_args}")

    model, tokenizer = setup_model_and_tokenizer(model_args)

    if training_args.gradient_checkpointing:
        # 在新的transformers版本中，当使用deepspeed时，需要这样显式地启用
        # 但Trainer内部逻辑也会处理，这里加一句日志更清晰
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    tokenized_dataset = process_data(data_args, tokenizer)
    
    if not tokenized_dataset:
        logger.error("Data processing returned an empty dataset. Exiting.")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("Trainer initialized. Starting training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    logger.info(f"Training finished. Final model and state saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()