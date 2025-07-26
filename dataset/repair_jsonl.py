# 文件名: repair_jsonl_v2.py (稳定修复版)

import json
import sys
import os
import re

def repair_jsonl_robust(original_path, repaired_path):
    """
    一个更健壮的修复脚本，专门解决两种已知问题：
    1. 跳过非JSON行（如注释）。
    2. 修复由未转义的双引号引起的JSON错误，特别是 ` "key": ""value"" ` 这种模式。
    """
    print("--- 稳定版修复脚本启动 ---")
    print(f"读取原始文件: {original_path}")

    if not os.path.exists(original_path):
        print(f"!!! 错误: 原始文件不存在: '{original_path}'")
        return

    repaired_lines_count = 0
    skipped_lines = []

    try:
        # 同时打开读和写文件
        with open(original_path, 'r', encoding='utf-8') as infile, \
             open(repaired_path, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile, 1):
                line = line.strip()

                # 1. 跳过空行和注释行
                if not line or line.startswith('#'):
                    skipped_lines.append((i, "注释或空行", line))
                    continue
                
                # 2. 尝试直接解析，如果本身就是有效的，直接写入
                try:
                    json.loads(line)
                    outfile.write(line + '\n')
                    repaired_lines_count += 1
                    continue
                except json.JSONDecodeError:
                    # 如果直接解析失败，说明行已损坏，尝试修复
                    pass

                # 3. 核心修复逻辑：针对 `"key": ""value""` 模式
                # 这个正则表达式会查找 `": "` 后面紧跟着的 `"`，并进行转义
                # 例如: "answer": ""三手烟"" -> "answer": "\"三手烟\""
                # 这个模式能精确匹配您数据中的错误
                fixed_line = re.sub(r'(": ")"(.*?)"', r'\1"\\"\2\\""', line)
                
                # 4. 验证修复后的行是否为有效JSON
                try:
                    json.loads(fixed_line)
                    outfile.write(fixed_line + '\n')
                    repaired_lines_count += 1
                except json.JSONDecodeError:
                    # 如果修复后仍然失败，说明是未知错误，跳过该行
                    skipped_lines.append((i, "无法自动修复的JSON错误", line))

        print("\n--- 修复完成 ---")
        print(f"成功处理并写入了 {repaired_lines_count} 行。")
        print(f"跳过了 {len(skipped_lines)} 行。")

        if skipped_lines:
            print("\n跳过的行详情:")
            for line_num, reason, content in skipped_lines:
                print(f"  - 行 {line_num} ({reason}): {content}")

        print(f"\n✅ 成功！已将修复后的数据保存到新文件: {repaired_path}")

    except Exception as e:
        print(f"\n!!! 在读写文件过程中发生严重错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python repair_jsonl_v2.py <原始文件路径>")
        print("脚本将生成一个带有 '_repaired' 后缀的新文件。")
        sys.exit(1)
        
    original_file_path = sys.argv[1]
    path_parts = os.path.splitext(original_file_path)
    repaired_file_path = f"{path_parts[0]}_repaired{path_parts[1]}"
    
    repair_jsonl_robust(original_file_path, repaired_file_path)