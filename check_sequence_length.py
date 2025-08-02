from transformers import AutoTokenizer
import json

# 只需要加载 tokenizer 和你的数据
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-7B-Instruct")
data_path = './lora_raw_output.jsonl'

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

def format_prompt(example):
    system_prompt = """
        你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题
        接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下<temperment></temperment>标签之间
        然后你需要分析婴儿当前气质特征对应的养育策略，我应该怎么样做，避免和婴儿自身性格冲突的做法。分析放在<strategy></strategy>之间
        以上分析放在<think></think>标签之间
        最后给家长当前建议怎么做，既要贴合婴儿气质，又要紧密关联用户的问题，给出针对性的而不是泛泛而谈的答案。以上放在<answer></answer>标签之间
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
        # {"role": "user", "content": example["question"]},
        # {"role": "assistant", "content": example["answer"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
token_lengths = []
for item in raw_data:
    formatted_text = format_prompt(item)["text"]
    # 我们只关心 input_ids 的长度
    length = len(tokenizer(formatted_text).input_ids)
    token_lengths.append(length)

# 排序并查看最长的样本
token_lengths.sort(reverse=True)
print(f"数据总数: {len(token_lengths)}")
print(f"最长样本的 Token 长度: {token_lengths[0]}")
print(f"Top 5 长度: {token_lengths[:5]}")
# 查看 95% 的数据所在的长度分位点
p95_index = int(len(token_lengths) * 0.05)
print(f"95% 的样本长度小于: {token_lengths[p95_index]}")