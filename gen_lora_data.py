# generate_dpo_data.py

import os
import json
import time
import requests

# --- 配置区 ---
API_KEY = "sk-8a19d21d1c89453bb6176a36a261f18b"
API_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
INPUT_FILE = "newtasks.jsonl"
OUTPUT_FILE = "lora_raw_output.txt"

# 指令模板
GENERATOR_PROMPT_TEMPLATE = """
你是一个育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案

---内置知识库：婴儿气质四分类与养育策略（基于托马斯与切斯理论）
核心理论框架 (Core Theoretical Framework)
基本原则: 婴儿气质是天生的行为倾向，没有“好”与“坏”之分，只有不同。它是孩子与世界互动的基础方式。
核心概念：拟合优度 (Goodness of Fit): 育儿成功的关键，不在于改变孩子的气质，而在于调整养育方式、环境和期望，来匹配孩子的天生气质。高匹配度能促进孩子健康发展；低匹配度则可能引发行为问题和亲子冲突。模型在生成建议时，必须始终围绕“提升拟合优度”这一核心目标。
四种分类: 基于纽约纵向研究，婴儿气质可归为四大类型：随和型(约40%)、困难型(约10%)、迟缓型(约15%)及混合型(约35%)。
1. 随和型 / 安乐型 (Easy Child)
气质特点 (Characteristics): 高节律性、高趋向性、高适应性、中低反应强度、积极心境。
养育建议与注意事项 (Parenting Advice & Key Considerations):
注意点1：避免因“省心”而忽视: 主动地、规律地提供情感交流。不哭闹不代表没有需求。
注意点2：肯定其温和的情感表达: 细心观察并及时回应他们的细微信号。
2. 困难型 / 磨娘精型 (Difficult Child)
气质特点 (Characteristics): 低节律性、高回避性、低适应性、高反应强度、偏消极心境。
养育建议与注意事项 (Parenting Advice & Key Considerations):
注意点1：父母的情绪稳定是第一关键: 避免陷入“孩子激烈哭闹 → 父母挫败 → 严厉管教 → 孩子更激烈”的恶性循环。
注意点2：建立高度可预测的结构化环境: “规律”是这类孩子的安全感来源。坚持严格、固定的每日作息。
3. 迟缓型 / 慢热型 (Slow-to-Warm-Up Child)
气质特点 (Characteristics): 偏低活动水平、初次反应为回避、适应性慢、反应强度弱。
养育建议与注意事项 (Parenting Advice & Key Considerations):
注意点1：绝对禁止强迫与催促: 给予充足的时间和空间。强迫最具破坏性。
注意点2：父母扮演“安全基地”和“桥梁”: 父母先愉快地与新环境互动，做安全示范。
注意点3：用积极语言重新定义特质: 将“胆小”解读为“谨慎”、“观察力强”。
4. 混合型 (Combination / Mixed Type)
气质特点 (Characteristics): 不具备一致性，呈现其他三种类型特征的混合。情境依赖性强。
养育建议与注意事项 (Parenting Advice & Key Considerations):
注意点1：采用“解构式”养育法: 不问“孩子是哪一型？”，而问“在‘这件事’上，孩子表现出了哪种特点？”
注意点2：灵活性和观察力是关键: 灵活地在不同养育“工具箱”中切换，实现动态的“拟合优度”。
---
家长的提问和婴儿的气质是{user_prompt}


请严格按照以下说明，生成回答：
`prompt`:user_prompt
`response`:
你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题
接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下，气质分析放在<temperment></temperment>标签之间
然后你需要分析婴儿当前气质特征对应的养育策略，我应该怎么样做，避免和婴儿自身性格冲突的做法。分析放在<strategy></strategy>之间
以上分析放在<think></think>标签之间

最后给家长当前建议怎么做，既要贴合婴儿气质，又要紧密关联用户的问题，给出针对性的而不是泛泛而谈的答案。以上放在<answer></answer>标签之间
"""

def get_raw_response(user_prompt):
    final_prompt = GENERATOR_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": final_prompt}],
        "temperature": 0.5,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{API_BASE_URL}/chat/completions",
                                 headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        choices = response_data.get('choices')

        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message')
            if message and isinstance(message, dict):
                content = message.get('content')
                if content:
                    return content
        return None
    except Exception as e:
        print(f"[!] API请求错误: {e}")
        return None

if __name__ == "__main__":
    print(f"从 {INPUT_FILE} 读取数据...")

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

        for i, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line.strip())
                prompt = task.get("prompt")
                if not prompt:
                    continue

                print(f"\n--- 正在处理任务 {i} ---")
                print(f"问题: {prompt}")

                raw_text = get_raw_response(prompt)

                if raw_text:
                    print("--- 模型返回原文 ---")
                    print(raw_text)
                    print("--------------------")
                    
                    outfile.write(json.dumps({
                        "prompt": prompt,
                        "response": raw_text
                    }, ensure_ascii=False) + "\n")
                    
                    print(f"[✔] 写入成功")
                else:
                    print(f"[✘] 无返回，跳过")

                time.sleep(2)

            except json.JSONDecodeError:
                print(f"[!] 第 {i} 行不是合法的 JSON，跳过。")

    print("\n所有任务处理完毕！")