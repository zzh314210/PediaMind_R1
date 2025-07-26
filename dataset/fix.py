import os

# --- 配置 ---
# 你的原始、有逗号的文件
input_file_path = './sftdata/sft_train_data.jsonl' 
# 修复后输出的新文件名
output_file_path = './sftdata/sft_train_data_processed.jsonl' 

print(f"Starting to fix file: {input_file_path}")

cleaned_lines = []
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        # 移除行首和行尾的空白字符，然后移除行尾可能存在的逗号
        cleaned_line = line.strip()
        if cleaned_line.endswith(','):
            cleaned_line = cleaned_line[:-1] # 移除最后一个字符（逗号）
        
        # 确保不是空行才添加
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in cleaned_lines:
        outfile.write(line + '\n')

print(f"Successfully created fixed file: {output_file_path}")
print(f"Total lines processed: {len(cleaned_lines)}")
print("\nIMPORTANT: Please update your training command to use the new fixed file!")
print(f"Example: --data_path \"{output_file_path}\"")