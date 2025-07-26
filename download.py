from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen-1.5B-Chat"  # Huggingface上模型名称示例，改成你想用的模型名
save_path = "./models/Qwen1.5-1.8B-Chat"

if not os.path.exists(save_path):
    os.makedirs(save_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"模型已下载并保存到 {save_path}")
