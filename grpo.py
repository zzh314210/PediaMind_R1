import os
import torch
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# ========== 配置 ==========
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_name = "./models/Qwen2.5-7B-Instruct-lora"
output_dir = "./models/Qwen2.5-7B-Instruct-lora-grpo"

# ========== 系统提示词 ==========
SYSPROMPT = """ 
你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案
你需要首先思考家长的提问，用自己的话复述家长正面临什么样的育婴问题
接着你需要分析婴儿的气质特征，并将婴儿的气质特征和行为关联起来分析，气质特征的分析放在<temperment></temperment>标签之间
然后你需要分析婴儿当前行为和他气质特征对应的养育策略，分析放在<strategy></strategy>之间
以上分析放在<think></think>标签之间
最后给家长当前建议怎么做，放在<answer></answer>标签之间
"""

# ========== 加载模型 & LoRA ==========
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map="auto"
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

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ========== 准备数据 ==========
dataset = Dataset.from_json("./dataset/grpodata/grpodataset.jsonl")

def preprocess(examples):
    return {
        "prompt": [
            {"role": "system", "content": SYSPROMPT},
            {"role": "user", "content": examples["prompt"]}
        ],
        "answer": ''  # 确保有answer字段
    }

dataset = dataset.map(preprocess)

# ========== 奖励函数 ==========
def my_reward_func(prompts, completions, answers = None, **kwargs) -> list[float]:
    scores = []
    for comp in completions:  # comp 是当前prompt生成的多个回答
        for _ in comp:  # 每个回答一个分数
            scores.append(random.uniform(0, 1))
    return scores

# ========== GRPO 配置 ==========
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name="qwen-baby-grpo",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,  # 每个prompt生成多个回答
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=3,
    save_steps=500,
    max_grad_norm=0.5,
    report_to="tensorboard"
)

# ========== 启动训练 ==========
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[my_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(output_dir)

print("✅ 训练完成并保存到:", output_dir)
