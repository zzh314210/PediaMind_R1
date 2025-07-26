# 文件名: generate_dpo_data.py (最终无错版)

import os
import json
import time
import requests
from tasks import TASKS

# --- 配置区 ---
API_KEY = "sk-8a19d21d1c89453bb6176a36a261f18b"
API_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
OUTPUT_FILE = "raw_output.txt"

# 指令模板
GENERATOR_PROMPT_TEMPLATE = """
# 角色
你是一位专门为AI训练生成高质量数据的专家。
# 任务
你的任务是根据提供的【内置知识库】、【用户提问】和【婴儿气质类型】，生成一个完整的DPO（Direct Preference Optimization）训练数据样本。这个样本必须是一个包含`prompt`、`chosen`和`rejected`三个键的JSON对象。
---
### 【内置知识库：婴儿气质四分类与养育策略（基于托马斯与切斯理论）】
#### ## 核心理论框架 (Core Theoretical Framework)
*   基本原则: 婴儿气质是天生的行为倾向，没有“好”与“坏”之分，只有不同。它是孩子与世界互动的基础方式。
*   核心概念：拟合优度 (Goodness of Fit): 育儿成功的关键，不在于改变孩子的气质，而在于调整养育方式、环境和期望，来匹配孩子的天生气质。高匹配度能促进孩子健康发展；低匹配度则可能引发行为问题和亲子冲突。模型在生成建议时，必须始终围绕“提升拟合优度”这一核心目标。
*   四种分类: 基于纽约纵向研究，婴儿气质可归为四大类型：随和型(约40%)、困难型(约10%)、迟缓型(约15%)及混合型(约35%)。
#### ### 1. 随和型 / 安乐型 (Easy Child)
*   气质特点 (Characteristics): 高节律性、高趋向性、高适应性、中低反应强度、积极心境。
*   养育建议与注意事项 (Parenting Advice & Key Considerations):
    *   注意点1：避免因“省心”而忽视: 主动地、规律地提供情感交流。不哭闹不代表没有需求。
    *   注意点2：肯定其温和的情感表达: 细心观察并及时回应他们的细微信号。
#### ### 2. 困难型 / 磨娘精型 (Difficult Child)
*   气质特点 (Characteristics): 低节律性、高回避性、低适应性、高反应强度、偏消极心境。
*   养育建议与注意事项 (Parenting Advice & Key Considerations):
    *   注意点1：父母的情绪稳定是第一关键: 避免陷入“孩子激烈哭闹 → 父母挫败 → 严厉管教 → 孩子更激烈”的恶性循环。
    *   注意点2：建立高度可预测的结构化环境: “规律”是这类孩子的安全感来源。坚持严格、固定的每日作息。
#### ### 3. 迟缓型 / 慢热型 (Slow-to-Warm-Up Child)
*   气质特点 (Characteristics): 偏低活动水平、初次反应为回避、适应性慢、反应强度弱。
*   养育建议与注意事项 (Parenting Advice & Key Considerations):
    *   注意点1：绝对禁止强迫与催促: 给予充足的时间和空间。强迫最具破坏性。
    *   注意点2：父母扮演“安全基地”和“桥梁”: 父母先愉快地与新环境互动，做安全示范。
    *   注意点3：用积极语言重新定义特质: 将“胆小”解读为“谨慎”、“观察力强”。
#### ### 4. 混合型 (Combination / Mixed Type)
*   气质特点 (Characteristics): 不具备一致性，呈现其他三种类型特征的混合。情境依赖性强。
*   养育建议与注意事项 (Parenting Advice & Key Considerations):
    *   注意点1：采用“解构式”养育法: 不问“孩子是哪一型？”，而问“在‘这件事’上，孩子表现出了哪种特点？”
    *   注意点2：灵活性和观察力是关键: 灵活地在不同养育“工具箱”中切换，实现动态的“拟合优度”。
---
### # 本次生成任务的输入
*   用户提问: "{user_question}"
*   婴儿气质类型: "{temperament_type}"
---
### # 输出要求
请严格按照以下说明，生成一个JSON对象：
1.  `prompt` (字符串):
    *   格式必须是："用户提问：{user_question}\\n宝宝气质类型：{temperament_type}"。
2.  `chosen` (字符串):
    *   必须包含 `<think>` 和 `<answer>` 两个部分。
    *   在 `<think>` 部分，必须使用流畅的自然语言展示推理过程：
        1.  首先，将婴儿的具体行为与该气质的核心特点进行明确的关联和解释。这部分分析**必须**用 `<temperment>` 标签包裹。
        2.  接着，基于这些特点，总结出对应的核心养育策略，并解释其有效性。这部分策略**必须**用 `<strategy>` 标签包裹。
        3.  最后，自然地过渡到将要给出的具体建议。
    *   在 `<answer>` 部分，提供具体、可操作、并且与`<think>`中策略相符的育儿建议。
3.  `rejected` (字符串):
    *   **必须**只包含 `<answer>` 部分，绝不能包含 `<think>` 思维链。
    *   `rejected`中的建议**必须是通用的、不基于特定气质分析的育儿建议**。它应该看起来合理，但质量低于`chosen`中基于深度气质分析得出的建议。
---
### # 最终输出（请直接生成这个JSON对象，不要有任何其他多余的文字）
"""

# --- API调用函数 (已修复) ---
def get_raw_response(user_question, temperament_type):
    final_prompt = GENERATOR_PROMPT_TEMPLATE.format(
        user_question=user_question,
        temperament_type=temperament_type
    )
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
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        choices = response_data.get('choices')
        
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message') 
            
            if message and isinstance(message, dict):
                content = message.get('content')
                if content:
                    return content
        
        print(f"  [!] API响应格式不符合预期。")
        print(f"  [!] 完整响应: {response_data}")
        return None
            
    except requests.exceptions.RequestException as e:
        print(f"  [!] API请求失败: {e}")
        return None
    except Exception as e:
        print(f"  [!] 发生未知错误: {e}")
        return None

if __name__ == "__main__":
    print(f"准备开始生成原始数据，共 {len(TASKS)} 个任务。")
    print(f"所有原始响应将写入到: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i, task in enumerate(TASKS):
            print(f"\n--- 正在处理任务 {i+1}/{len(TASKS)} ---")
            print(f"问题: {task['question']}")
            print(f"气质: {task['temperament']}")
            
            raw_text = get_raw_response(task['question'], task['temperament'])
            
            if raw_text:
                print("--- 模型返回原文 ---")
                print(raw_text)
                print("--------------------")
                
                f.write(raw_text + "\n")
                f.write("---END OF RESPONSE---\n\n")
                
                print(f"  [✔] 成功获取响应并写入到 {OUTPUT_FILE}")
            else:
                print(f"  [✘] 获取响应失败，跳过。")
            
            time.sleep(2) 

    print("\n所有任务处理完毕！")