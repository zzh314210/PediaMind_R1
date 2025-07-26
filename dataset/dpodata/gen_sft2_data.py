import json

def convert_dpo_to_sft_no_clean(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                chosen = data.get('chosen', '')
                sft_item = {
                    "question": prompt,
                    "answer": chosen
                }
                outfile.write(json.dumps(sft_item, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"跳过无法解析的行: {line[:80]}... 错误: {e}")

# 用法示例
convert_dpo_to_sft_no_clean('clean_dpo_data.jsonl', 'sft_data2.jsonl')
