import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "models/Qwen2.5-7B-Instruct"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, CUDA device count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)

    model.eval()

    while True:
        text = input("Input your query (type 'exit' to quit): ")
        if text.strip().lower() == "exit":
            break
        prompt = f"<|im_start|>system\n你是一个简洁严谨的助手，请只回答用户提出的问题，不进行拓展。\n<|im_end|>\n<|im_start|>user\n{text.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        # 从输入部分之后截取生成内容
        generated = outputs[0][inputs['input_ids'].size(1):]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
