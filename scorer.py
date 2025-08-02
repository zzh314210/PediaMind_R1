# 文件名: scorer.py

import os
import re
import json
import time
import requests
import logging
from typing import List, Dict

# --- API配置 ---
# 建议使用环境变量来管理API密钥，这样更安全。
# 在运行训练前，在终端执行: export UNIDT_API_KEY="你的sk密钥"
# 如果不使用环境变量，也可以直接在这里修改默认值。
SK_KEY = os.environ.get("UNIDT_API_KEY", "sk-4051b5b1088b4ef3a2e7bed50d2edb6f") 
API_URL = "https://ai-api.unidtai.com/openapi/llm/chat"
API_MODEL = "DeepSeek-V3" # 您用于评分的模型

# --- 日志配置 ---
# 日志格式与主训练脚本保持一致，方便统一查看
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RewardScorer")


def construct_scoring_prompt(user_prompt: str, completion: str) -> str:
    """
    根据论文中定义的三个维度，构建一个复合的评分Prompt。
    这个Prompt将指导大模型一次性完成所有评估并计算总分。
    
    Args:
        user_prompt (str): 原始的用户提问。
        completion (str): SFT模型生成的回答。
        
    Returns:
        str: 完整的评分Prompt。
    """
    # 从 user_prompt 中安全地提取气质标签，如果找不到则为"未知"
    temperament = "未知"
    temperament_match = re.search(r"宝宝气质：(困难型|迟缓型|随和型|混合型)", user_prompt)
    if temperament_match:
        temperament = temperament_match.group(1)

    return f"""
    你是一位育婴专家和AI模型评估员。请根据以下三个维度，对AI助手的回答进行严格评分，并计算总分。
首先你需要了解婴儿气质四分类与养育策略（基于托马斯与切斯理论）
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

    接下来你需要根据家长的问题，对AI助手的回答打分
    家长的问题:{user_prompt}
    AI助手的回答:{completion}
评分标准如下：
1. 格式规范性 (R_fmt, 满分1分):
   回答是否严格包含 <think>, <answer>, <temperment>, <strategy>这四个标签？如果严格满足给1分，否则给0分

2. 气质对齐度 (R_temp, 满分1分):
家长的宝宝气质是 {temperament}。
你需要判断<temperament>标签内的气质分析是否符合婴儿气质四分类与养育策略，符合给0.5分
你还需要判断气质是否结合了行为分析并安抚家长，如果有气质结合行为分析就给0.5分
   

3. 知识相关性 (R_know, 满分1分):
<answer>标签中的最终建议是否专业、可操作，和家长的提问紧密关联，能有效解决家长的核心问题？如果是给1分，如果不是给0分

请计算总分 R(y) = R_fmt + R_temp + R_know。
满分示例如下：
示例1：
prompt:"我的宝宝对常规的作息安排有极强的抵抗情绪，任何小变动都会引发长时间的哭闹，该如何是好？宝宝气质：困难型"
response:
<think> 家长面临的问题是：宝宝对日常作息安排表现出强烈的抵触情绪，即使是微小的变动也会引发长时间的哭闹，这让家长感到困扰和不知所措。<temperment> 根据描述，宝宝属于困难型气质。困难型婴儿通常具有低节律性、高回避性、低适应性、高反应强度和偏消极心境的特点。这意味着他们对变化非常敏感，适应新事物或变化需要更多时间，且情绪反应强烈。家长无需过度担心，这是宝宝天生的气质特点，并非故意为之或家长养育不当。 </temperment><strategy> 针对困难型婴儿，养育策略应着重于： 1. 建立高度可预测的结构化环境：规律性是这类孩子的安全感来源。 2. 保持父母情绪稳定：避免因孩子的强烈反应而情绪波动，形成恶性循环。 3. 逐步引入变化：任何改变都应提前预告并分步实施。 4. 坚持一致的作息：即使孩子抵抗，也要温和但坚定地维持常规。 避免的做法： - 频繁改变作息安排 - 在孩子哭闹时妥协或改变计划 - 表现出焦虑或愤怒情绪 </strategy> </think><answer> 针对您的情况，建议采取以下具体措施：
建立铁打不变的作息表
制定详细的每日作息（进食、睡眠、活动时间），精确到15分钟误差内
用大型可视化日历展示作息，每次活动前温和提醒："现在是午睡时间了"
变化引入的"三步缓冲法"
提前预告：在改变前1小时开始提醒（"再过1小时我们要洗澡了"）
分步过渡：将变化拆解为3个小步骤（如：先展示浴盆→再放玩具→最后脱衣）
事后复盘：结束后立即肯定（"刚才洗澡你很配合"）
哭闹时的"锚定回应"
保持平静表情和60分贝以下的音量
重复固定安抚短语："妈妈在这里，现在是XX时间"
提供"过渡物"（同一块安抚巾/特定玩具）
环境微调技巧
在作息交替处设置明显视觉标记（如餐椅铺固定颜色的餐垫）
使用气味提示（固定品牌的沐浴露/餐前擦同款护手霜）
请记录1-2周的宝宝反应模式，通常第3周开始会观察到抵抗强度降低。困难型孩子对规律性有惊人记忆力，一旦建立可靠常规，反而会成为最坚守规则的人。
</answer>
你应该返回3.0

返回格式:
请只返回最终计算出的总分（一个小数），不要包含任何其他文字、解释或理由。
例如: 2.5
"""

def call_llm_api(prompt: str) -> str:
    """
    调用大模型API并返回文本结果，包含重试机制。
    
    Args:
        prompt (str): 发送给大模型的完整Prompt。
        
    Returns:
        str: 大模型的原始文本响应。
    """
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # 零温度确保评分的稳定性
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SK_KEY}"
    }

    # 增加简单的重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.post(API_URL, headers=headers, json=payload, timeout=(10, 120)) as response:
                response.raise_for_status() # 如果状态码不是2xx，则抛出异常
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # 指数退避等待
            else:
                return "[ERROR] API_CALL_FAILED_AFTER_RETRIES"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"API响应解析失败: {e}, 响应内容: {response.text}")
            return "[ERROR] API_RESPONSE_PARSING_FAILED"
    
    return "[ERROR] UNKNOWN_API_ERROR"


def parse_score(llm_response: str) -> float:
    """
    从大模型的响应中解析出最终分数。
    
    Args:
        llm_response (str): 大模型的原始文本响应。
        
    Returns:
        float: 解析出的分数，范围在0.0到3.0之间。
    """
    # 首先检查是否有错误标识
    if llm_response.startswith("[ERROR]"):
        logger.warning(f"API调用返回错误: {llm_response}，返回0分。")
        return 0.0
        
    try:
        # 使用正则表达式匹配响应中的第一个数字（整数或小数）
        # 这个表达式更健壮，可以处理 "2.5", "得分: 2.5", "2.5分" 等情况
        score_match = re.search(r'(\d\.?\d*)', llm_response)
        
        if score_match:
            score = float(score_match.group(1))
            # 确保分数在论文定义的合理范围内 (0.0 到 1+1+1=3.0)
            return max(0.0, min(3.0, score))
        else:
            logger.warning(f"无法从响应 '{llm_response}' 中提取数字分数，返回0分。")
            return 0.0
    except (ValueError, TypeError):
        logger.warning(f"从响应 '{llm_response}' 中解析浮点数失败，返回0分。")
        return 0.0

def get_user_prompt_from_conversation(conversation: List[Dict]) -> str:
    """
    从TRL格式的对话历史中安全地提取最后一条用户的问题。
    
    Args:
        conversation (List[Dict]): 对话历史列表。
        
    Returns:
        str: 用户问题的文本内容，如果找不到则返回空字符串。
    """
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "")
    return ""

def score_completions(prompts: List[List[Dict]], completions: List[str]) -> List[float]:
    """
    主函数：接收批量数据，为每个completion评分。
    这是训练脚本 `train_grpo.py` 将要调用的核心函数。
    
    Args:
        prompts (List[List[Dict]]): TRL格式的对话提示列表。
        completions (List[str]): 模型生成的回答列表。
        
    Returns:
        List[float]: 每个回答对应的复合奖励分数列表。
    """
    batch_scores = []
    
    if len(prompts) != len(completions):
        logger.error("Prompts和Completions的数量不匹配！将为这个批次返回全0分数。")
        return [0.0] * len(completions)

    for i in range(len(completions)):
        user_prompt = get_user_prompt_from_conversation(prompts[i])
        completion = completions[i]

        if not user_prompt:
            logger.warning(f"样本 {i} 中未找到有效的用户prompt，评分为0。")
            batch_scores.append(0.0)
            continue
        
        # 1. 构建评分用的Prompt
        scoring_prompt = construct_scoring_prompt(user_prompt, completion)
        
        # 2. 调用API获取评分结果
        llm_response = call_llm_api(scoring_prompt)
        
        # 3. 解析分数
        score = parse_score(llm_response)
        batch_scores.append(score)
        
        # 在训练时，主进程会打印更详细的日志，这里仅为独立测试提供信息
        # logger.info(f"样本 {i+1}/{len(completions)} | 分数: {score} (LLM response: '{llm_response}')")

    return batch_scores


# --- 独立测试入口 ---
# 运行 `python scorer.py` 可以测试这个脚本的功能是否正常
if __name__ == "__main__":
    # logger.info("="*20 + " 开始独立测试 scorer.py (API连通性测试) " + "="*20)
    
    # # 1. 准备一个最简单的测试Prompt
    # test_prompt = "你好，请介绍一下你自己。"
    
    # # 2. 直接调用API函数
    # logger.info(f"正在向API发送请求，内容: '{test_prompt}'")
    # api_response = call_llm_api(test_prompt)
    
    # # 3. 打印结果
    # print("\n" + "="*24 + " 测试结果 " + "="*24)
    # if api_response.startswith("[ERROR]"):
    #     print(f"❌ API调用失败！")
    #     print(f"   错误信息: {api_response}")
    # else:
    #     print(f"✅ API调用成功！")
    #     print(f"   模型返回内容: {api_response}")
    # print("="*58)
    logger.info("="*20 + " 开始独立测试 scorer.py " + "="*20)
    test_prompts_data = [
    # Case 1: 完美回答
    [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '我的孩子似乎对别人的负面情绪有很强的‘消化’能力，我该如何让他学会适度地表达自己的不满？宝宝气质：随和型'}],
    # Case 2: 格式错误
    [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '为什么孩子有时能表现出惊人的社交直觉？宝宝气质：混合型'}],
    # Case 3: 气质分析错误
    [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '我儿子对新环境的适应过程就像电影里的慢动作。宝宝气质：迟缓型'}],
]
    test_completions_data = [
        # Case 1 的回答: 格式正确，气质对齐，知识相关 (预期得分: 1+1+1 = 3.0)
        "<think><temperment>随和型宝宝天生倾向于避免冲突，可能会为了维持和谐而压抑自己的不满情绪。</temperment><strategy>关键是创造一个安全的、鼓励表达真实感受的环境。家长需要主动引导并验证他的负面情绪，让他明白表达不满也是可以的。</strategy></think><answer>您可以在日常小事中主动问他：‘你是不是不喜欢这个？告诉妈妈没关系。’ 当他表达时，立刻给予积极反馈：‘谢谢你告诉我你的真实想法，妈妈很高兴。’ 这样能让他慢慢建立表达的信心。</answer>",
        # Case 2 的回答: 缺少 <think> 和 <temperment> 标签 (预期得分: 0+1+0.5 = 1.5, 假设气质和知识部分相关)
        "<strategy>混合型宝宝就是这样，行为多变，有时很敏锐。</strategy><answer>家长平常心看待就好，多观察孩子的行为模式，在他读不懂空气时温和提醒。</answer>",
        # Case 3 的回答: 气质分析错误，把迟缓型分析成了困难型 (预期得分: 1+0+0.5 = 1.5)
        "<think><temperment>困难型宝宝天生就很固执，对新环境反应激烈。</temperment><strategy>需要家长用坚定的态度来引导。</strategy></think><answer>家长可以先让他自己待一会儿，等他情绪平复了再引导他参与。</answer>",
    ]

    final_scores = score_completions(test_prompts_data, test_completions_data)

    print("\n" + "="*22 + " 测试结果 " + "="*22)
    for i in range(len(final_scores)):
        print(f"--- 测试用例 {i+1} ---")
        print(f"用户问题: {test_prompts_data[i][-1]['content']}")
        # print(f"模型回答: {test_completions_data[i]}")
        print(f"大模型评分: {final_scores[i]}")
    print("="*56)