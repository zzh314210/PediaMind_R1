import json
import sys
import os

def validate_jsonl_debug(file_path):
    # 1. 打印入口信息和参数
    print("--- 调试脚本启动 ---")
    print(f"Python 版本: {sys.version}")
    print(f"脚本接收到的文件路径: {file_path}")

    # 2. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"!!! 错误: 文件路径不存在: '{file_path}'")
        print("--- 调试脚本结束 ---")
        return

    print(f"文件存在。正在检查文件大小...")
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")

    if file_size == 0:
        print("!!! 警告: 文件是空的。")
        print("--- 调试脚本结束 ---")
        return

    # 3. 尝试打开并读取文件
    invalid_lines = []
    print("准备进入循环，逐行读取文件...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print("文件成功打开，开始循环。")
            for i, line in enumerate(f, 1):
                # 在循环内部，检查每一行
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    # 发现错误，立即打印并记录
                    print(f"!!! 在第 {i} 行发现JSON错误: {e}")
                    invalid_lines.append((i, str(e), line.strip()))
            
            # 如果循环正常结束
            print(f"文件读取完毕，共处理 {i} 行。")

    except Exception as e:
        # 捕捉打开或读取文件时可能发生的任何其他异常
        print(f"!!! 在文件处理过程中发生未知错误: {e}")
        print("--- 调试脚本结束 ---")
        return

    # 4. 报告最终结果
    print("\n--- 最终检查结果 ---")
    if not invalid_lines:
        print(f"✅ 文件 '{file_path}' 中的所有行都是有效的 JSON。")
    else:
        print(f"❌ 在文件 '{file_path}' 中发现 {len(invalid_lines)} 个无效的JSON行：")
        for line_num, error_msg, line_content in invalid_lines:
            print(f"  - 行 {line_num}: {error_msg}")
            print(f"    内容: {line_content}\n")
    
    print("--- 调试脚本结束 ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python fix2.py <path_to_your_file.jsonl>")
        sys.exit(1)
    
    file_to_check = sys.argv[1]
    validate_jsonl_debug(file_to_check)