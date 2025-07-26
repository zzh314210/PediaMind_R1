import json

def extract_dpo_data(raw_text):
    result = []
    current = {}  # 当前正在构造的 json 对象
    lines = raw_text.replace('\\n', ' ').replace('\\', ' ').splitlines()  # 去掉 \n 后按行处理

    for line in lines:
        line = line.strip()
        if line.startswith('```json'):
            current = {}  # 开始一个新条目
        elif line.startswith('"prompt"'):
            key, value = line.split(":", 1)
            current["prompt"] = value.strip().strip('",')
        elif line.startswith('"chosen"'):
            key, value = line.split(":", 1)
            current["chosen"] = value.strip().strip('",')
        elif line.startswith('"rejected"'):
            key, value = line.split(":", 1)
            current["rejected"] = value.strip().strip('",')
        elif line.startswith('---END OF RESPONSE---'):
            if all(k in current for k in ("prompt", "chosen", "rejected")):
                result.append(current)
                current = {}

    return result

# 示例输入
with open("raw_output.txt", "r", encoding="utf-8") as f:
    raw_text_data = f.read()

# 提取数据
dpo_dataset = extract_dpo_data(raw_text_data)

# 保存为 JSONL 文件
with open("clean_dpo_data.jsonl", "w", encoding="utf-8") as f:
    for entry in dpo_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
