import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_PATH = "./models/Qwen2.5-7B-Instruct-sft-grpo"

def main():
    """
    一个支持结构标签输出的交互式聊天脚本
    """
    print("正在加载模型和分词器...")
    print(f"模型路径: {MODEL_PATH}")

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
        sys.exit(1)

    print("\n模型加载完毕！现在可以开始对话了。")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("输入 'clear' 清空对话历史。")
    print("-" * 30)

    # ✅ 系统提示词，强制要求结构化输出
    system_prompt = {
        "role": "system",
        "content": (
        """
        # 你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案
        # 你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题
        # 接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下，气质分析放在<temperment></temperment>标签之间
        # 然后你需要分析婴儿当前气质特征对应的养育策略，我应该怎么样做，避免和婴儿自身性格冲突的做法。养育策略放在<strategy></strategy>之间
        # 以上所有分析放在<think></think>标签之间
        # 最后给家长当前建议怎么做，既要贴合婴儿气质，又要紧密关联用户的问题，给出针对性的而不是泛泛而谈的答案。回答完以后再简短总结，给家长信心。以上放在<answer></answer>标签之间
        """
        # """
        # 你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行合理的分析并选出最合理的选择题答案
        # 你需要首先思考家长的提问，家长正面临什么样的育婴问题
        # 接着你需要分析婴儿的气质特征
        # 然后你需要分析婴儿当前气质特征对应的养育策略，几个选项是否合适，避免和婴儿自身性格冲突的做法。
        # 最后给出选择题的唯一答案，只返回答案
        # """
        # """
        # 你是一个育婴专家，你的任务是回答家长的育婴提问，选出最合理的选择题答案
        # 你要给出选择题答案，并且只返回答案
        # """
        )
    }

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

            messages = [system_prompt] + history + [{"role": "user", "content": query}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = model_inputs["input_ids"].shape[1]
            output = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
            print(f"Assistant: {output}")

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": output})

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
