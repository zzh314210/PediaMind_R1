import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_PATH = "./models/Qwen2.5-7B-Instruct-sft-sft-dpo"

def main():
    """
    一个简单的交互式聊天脚本
    """
    print("正在加载模型和分词器...")
    print(f"模型路径: {MODEL_PATH}")

    # 当使用 device_map="auto" 时，不需要手动指定 device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"使用的设备: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

    except Exception as e:
        print(f"加载模型或分词器时出错: {e}")
        print("请确保路径正确，并且模型文件完整。")
        sys.exit(1)

    print("\n模型加载完毕！现在可以开始对话了。")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("输入 'clear' 清空对话历史。")
    print("-" * 30)

    history = []

    while True:
        try:
            query = input("You: ")

            if query.lower() in ["quit", "exit"]:
                print("再见！")
                break
            
            if query.lower() == "clear":
                history = []
                print("对话历史已清空。")
                print("-" * 30)
                continue

            messages = history + [{"role": "user", "content": query}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt") 
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
            # ==========================================================

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            input_ids_len = model_inputs["input_ids"].shape[1]
            response_ids = generated_ids[0][input_ids_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            print(f"Assistant: {response}")
            
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()