# train_lora.py (最终修正版)

import os
import json
import torch
import datetime
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    HfArgumentParser
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

# ========== 日志设置 ==========
if int(os.environ.get("RANK", 0)) == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)

# ========== 参数定义 ==========
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "预训练模型的路径或Hugging Face模型标识符"})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "训练数据 .jsonl 文件的路径"})
    max_seq_length: int = field(default=1024, metadata={"help": "Tokenize后的最大序列长度"})

# ========== 显存监控回调 ==========
def log_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"[{prefix}] 💾 GPU显存 -> 已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB")

class MemoryLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and state.global_step % args.logging_steps == 0:
            log_memory(f"Step {state.global_step}")

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        logger.info(f"正在加载模型: {model_args.model_name_or_path}")
        logger.info(f"所有训练参数: {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if training_args.local_rank == 0:
        logger.info("正在配置 LoRA ...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    model.enable_input_require_grads()
    if training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.local_rank == 0:
        logger.info(f"正在从 {data_args.data_path} 加载数据...")
        
    with open(data_args.data_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    def format_prompt(example):
        system_prompt = """
            你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案
            你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题
            接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下，气质分析放在<temperment></temperment>标签之间
            然后你需要分析婴儿当前气质特征对应的养育策略，我应该怎么样做，避免和婴儿自身性格冲突的做法。养育策略放在<strategy></strategy>之间
            以上所有分析放在<think></think>标签之间
            最后给家长当前建议怎么做，既要贴合婴儿气质，又要紧密关联用户的问题，给出针对性的而不是泛泛而谈的答案。回答完以后再简短总结，给家长信心。以上放在<answer></answer>标签之间
            """
        messages = [
            {"role": "system", "content": system_prompt},
            # {"role": "user", "content": example["question"]},
            # {"role": "assistant", "content": example["answer"]}
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list([format_prompt(x) for x in raw_data])

    # ==================== 函数修正区域 V3 ====================
    def tokenize_fn(examples):
        im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
        
        # 核心改动：重新启用在 map 阶段的填充。
        # 这确保了送入 DataCollator 之前，每个样本的 'input_ids', 'attention_mask', 
        # 和我们手动创建的 'labels' 都具有相同的长度。
        tokenized_output = tokenizer(
            examples["text"],
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length", # <-- 改回 "max_length"
        )
        input_ids_list = tokenized_output["input_ids"]
        labels_list = []

        for input_ids in input_ids_list:
            # 现在 input_ids 已经是被填充过的列表了
            labels = list(input_ids)
            assistant_start_index = -1
            try:
                # 找到 assistant 部分的起始点
                # 从1开始循环，避免 input_ids[i-1] 越界
                for i in range(1, len(input_ids)):
                    if input_ids[i-1] == im_start_token_id and input_ids[i] == assistant_token_id:
                        assistant_start_index = i - 1 # assistant部分的起始位置是 <|im_start|>
                        break
            except IndexError:
                # 在极少数情况下，如果序列被截断得非常短，可能会发生索引错误
                assistant_start_index = -1

            if assistant_start_index != -1:
                # 屏蔽 assistant 之前的所有内容
                for i in range(assistant_start_index):
                    labels[i] = -100
            else:
                # 安全措施: 如果没找到，屏蔽所有标签
                if training_args.local_rank == 0:
                    logger.warning(
                        "在一条数据中未找到 assistant 标识，将屏蔽所有标签。请检查数据格式或截断长度。"
                        f" (序列长度: {len(input_ids)})"
                    )
                for i in range(len(labels)):
                    labels[i] = -100
            
            labels_list.append(labels)
        
        tokenized_output["labels"] = labels_list
        return tokenized_output
    # =========================================================

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=os.cpu_count() // 2)
    if training_args.local_rank == 0:
        logger.info(f"数据处理完成，数据集大小: {len(tokenized_dataset)}")
        logger.info("数据样本预览 (列):")
        print(tokenized_dataset.column_names)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[MemoryLoggerCallback()]
    )

    if training_args.local_rank == 0:
        logger.info(f"🚀 训练开始！将在 {torch.cuda.device_count()} 张GPU上进行训练。")
    
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # if training_args.local_rank == 0:
    #     logger.info(f"✅ 训练完成，正在保存 LoRA adapter 到: {training_args.output_dir}")
    #     model.save_pretrained(training_args.output_dir)
    #     tokenizer.save_pretrained(training_args.output_dir)
    #     logger.info("模型保存完毕！")

    if training_args.local_rank == 0:
        logger.info(f"✅ 训练完成，正在合并 LoRA 权重并保存合并后的模型到: {training_args.output_dir}")
        merged_model = model.merge_and_unload()  # 合并 LoRA 到基础模型
        merged_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("合并模型保存完毕！")

if __name__ == "__main__":
    main()