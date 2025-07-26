import json
import argparse
import os
import sys

def sort_jsonl_by_length(input_path: str, output_path: str):
    """
    Reads a JSONL file, sorts it by the combined length of 'question' and 'answer' fields,
    and writes the sorted data to a new JSONL file.
    """
    print(f"Reading data from: {input_path}")
    
    data_with_lengths = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    # 确保 'question' 和 'answer' 字段存在
                    if 'question' not in item or 'answer' not in item:
                        print(f"Warning: Skipping line {i+1} because it's missing 'question' or 'answer' field.", file=sys.stderr)
                        continue
                        
                    # 计算总长度（问题+答案），这是一个很好的长度代理指标
                    total_length = len(item['question']) + len(item['answer'])
                    data_with_lengths.append({'length': total_length, 'data': item})
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {i+1} due to JSON decoding error.", file=sys.stderr)
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.", file=sys.stderr)
        sys.exit(1)

    if not data_with_lengths:
        print("Error: No valid data was loaded. The input file might be empty or in the wrong format.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully loaded {len(data_with_lengths)} items. Sorting now...")
    
    # 按长度排序
    sorted_data = sorted(data_with_lengths, key=lambda x: x['length'])

    print(f"Sorting complete. Writing to: {output_path}")
    
    # 将排序后的数据写回新的jsonl文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sorted_data:
                f.write(json.dumps(item['data'], ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing to file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Success! Data sorted by length and saved to {output_path}")


if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="Sort a JSONL file based on the length of its 'question' and 'answer' fields.",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    
    # 添加一个必需的位置参数 'input_file'
    parser.add_argument(
        'input_file', 
        type=str, 
        help="Path to the input JSONL file.\nExample: python sort_jsonl.py ./data/my_data.jsonl"
    )

    args = parser.parse_args()

    # --- 逻辑处理 ---
    input_path = args.input_file
    
    # 从输入路径生成输出路径
    # 1. 获取目录名
    directory = os.path.dirname(input_path)
    # 2. 获取不带扩展名的文件名
    base_name = os.path.basename(input_path)
    filename, ext = os.path.splitext(base_name)
    # 3. 构造新的文件名
    output_filename = f"{filename}_sorted{ext}"
    # 4. 组合成最终的输出路径
    output_path = os.path.join(directory, output_filename)
    
    # 调用主函数
    sort_jsonl_by_length(input_path, output_path)