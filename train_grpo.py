# 文件名: train_grpo.py
# ---------------------------------------------
import os
import torch
import random
import logging
from dataclasses import dataclass, field
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import scorer
# 只在主进程上配置日志
if int(os.environ.get("RANK", 0)) == 0:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 参数定义 ==========
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "预训练SFT模型的路径"})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "GRPO 训练数据 .jsonl 文件的路径"})

# ========== 系统提示词 (与您脚本完全一致) ==========
SYSPROMPT = """ 
            你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案
            你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题
            接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下，气质分析放在<temperment></temperment>标签之间
            然后你需要分析婴儿当前气质特征对应的养育策略，我应该怎么样做，避免和婴儿自身性格冲突的做法。养育策略放在<strategy></strategy>之间
            以上所有分析放在<think></think>标签之间
            最后给家长当前建议怎么做，既要贴合婴儿气质，又要紧密关联用户的问题，给出针对性的而不是泛泛而谈的答案。回答完以后再简短总结，给家长信心。以上放在<answer></answer>标签之间
"""

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        logger.info(f"所有训练参数: {training_args}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    if training_args.local_rank == 0:
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    # ========== 核心修正：在这里增加数据过滤 ==========
    # 过滤掉 "prompt" 字段为空 (None) 或者为空字符串的数据
    original_size = len(dataset)
    # 使用 .get() 方法安全地访问字段，避免 KeyErorr
    dataset = dataset.filter(lambda example: example.get("prompt") is not None and len(example["prompt"].strip()) > 0)
    if training_args.local_rank == 0:
        filtered_size = len(dataset)
        logger.info(f"数据过滤完成：原始数据量 {original_size}, 过滤后数据量 {filtered_size}。移除了 {original_size - filtered_size} 条无效数据。")
    # ================================================


    def preprocess(examples):
        # 注意：这里的 examples 已经是过滤后的，所以 examples["prompt"] 不会是 None
        return {
            "prompt": [
                {"role": "system", "content": SYSPROMPT},
                {"role": "user", "content": examples["prompt"]}
            ],
            "answer": ''
        }
    
    dataset = dataset.map(preprocess, remove_columns=list(dataset.features))

    # def my_reward_func(prompts, completions, answers = None, **kwargs) -> list[float]:
    #     return [random.uniform(0, 1) for _ in completions]
    def my_reward_func(prompts, completions, answers = None, **kwargs) -> list[float]: # <--- 删掉前面的空格，和上面的 preprocess 函数对齐
        """
        使用大模型API评分的奖励函数，并支持分布式训练。
        """
        # 约定只在 rank 0 的进程上进行API评分，避免资源浪费和API超限
        scores = []
        if training_args.local_rank == 0:
            logger.info(f"Step {kwargs.get('step', 'N/A')}: Rank 0 开始调用大模型进行评分...")
            # 调用我们独立脚本中的评分函数
            scores = scorer.score_completions(prompts, completions)
            logger.info(f"Step {kwargs.get('step', 'N/A')}: Rank 0 评分完成，获得分数: {scores}")

        # 使用PyTorch的分布式通信来广播(broadcast)分数
        # 这样所有GPU进程都能拿到同样的分数列表
        if torch.distributed.is_initialized():
            # 将分数列表转换为Tensor，以便进行广播
            # 注意：需要确保scores在非rank 0进程上是一个空列表，否则tensor尺寸不匹配会报错
            # 但由于broadcast会用rank 0的数据覆盖，所以问题不大，但规范起见可以初始化
            if training_args.local_rank != 0:
                # 创建一个形状和类型都匹配的空tensor用于接收广播
                scores_tensor = torch.empty(len(completions), device=training_args.device, dtype=torch.bfloat16)
            else:
                scores_tensor = torch.tensor(scores, device=training_args.device, dtype=torch.bfloat16)

            torch.distributed.broadcast(scores_tensor, src=0)
            
            # 所有进程从Tensor中获取最终的分数列表
            final_scores = scores_tensor.cpu().tolist()
        else:
            # 如果不是分布式环境，直接使用 rank 0 的分数
            final_scores = scores
            
        return final_scores

    # ========== 启动训练 ==========
    trainer = GRPOTrainer(
        model=model,
        # ==================== 核心修正 ====================
        # 将 'tokenizer' 改回您环境中兼容的 'processing_class'
        processing_class=tokenizer,
        # ================================================
        reward_funcs=[my_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    if training_args.local_rank == 0:
        logger.info("🚀 GRPO 训练开始！")

    trainer.train()

    if training_args.local_rank == 0:
        logger.info("✅ 训练完成，开始合并LoRA权重并保存完整模型...")

        # 1. 从 PeftModel 中获取基础模型，并自动合并LoRA权重
        merged_model = model.merge_and_unload()
        logger.info("LoRA权重合并完成。")

        # 2. 保存完整模型
        # 这会像保存普通Hugging Face模型一样，保存所有文件（包括模型权重、配置文件等）
        merged_model.save_pretrained(training_args.output_dir)
        logger.info(f"完整模型已保存到: {training_args.output_dir}")

        # 3. 同时保存Tokenizer
        # 这样加载模型时会更方便，确保模型和Tokenizer是配套的
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer已保存到: {training_args.output_dir}")
        
        logger.info("🎉 所有保存操作完成！")

if __name__ == "__main__":
    main()