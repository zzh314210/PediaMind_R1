import torch
from transformers import AutoTokenizer

# ========== 1. 配置和加载 ==========
# 和你的主脚本保持一致
model_path = "./models/Qwen2.5-7B-Instruct" 
print(f"正在从 '{model_path}' 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer 加载完成。")

# ========== 2. 准备单条测试数据 ==========
one_shot = {
    "prompt": "为什么孩子在需要创造性思维时表现平平，但在需要批判性思维时却很敏锐？宝宝气质：混合型",
    "response": "<think>\n家长面临的问题是：观察到孩子在需要创造性思维（如开放式艺术活动、想象游戏）时表现较为平淡，但在需要批判性思维（如解决问题、分析规则）时却展现出敏锐度。这其实反映了混合型气质孩子的典型特征——在不同情境下会表现出不同气质倾向的复合反应。\n\n<temperment>\n混合型气质的孩子没有单一的行为模式，他们的表现高度依赖具体情境。您描述的现象非常正常：\n1. 批判性思维敏锐：可能体现了迟缓型气质中的\"谨慎观察\"特质，或随和型气质中\"适应性高\"的特点\n2. 创造性表现平平：可能反映困难型气质对无序情境的本能回避，或迟缓型需要更长时间预热的特点\n请放心，这绝不意味着孩子缺乏创造力，只是表达方式可能不同。\n</temperment>\n\n<strategy>\n针对混合型气质的\"解构式养育\"策略：\n1. 情境分析法：区分孩子在不同创造性活动中的具体反应（如对自由绘画vs结构化手工的不同表现）\n2. 渐进引导法：先提供有框架的创作（如填色游戏），再逐步过渡到开放式创作\n3. 多通道激发：通过孩子擅长的批判性活动（如积木搭建）自然延伸出创造性表达\n需避免：\n× 贴\"缺乏创意\"标签\n× 强行推进完全自由创作\n× 忽视孩子通过其他形式（如语言描述）展现的创造力\n</strategy>\n</think>\n\n<answer>\n建议采取以下具体方法帮助孩子发展创造性思维：\n\n1. 搭建\"创造性脚手架\"：\n- 先从孩子擅长的分析性活动切入（如：\"我们来研究怎么用这些积木搭得更高\"）\n- 逐步引入开放元素（如：\"如果加个斜坡，你觉得会发生什么有趣的事？\"）\n\n2. 创造安全表达环境：\n- 准备半结构化艺术材料（如：有轮廓线的画纸+自由装饰区）\n- 采用\"1+1\"原则：每次创作包含1个指定元素+1个自选元素\n\n3. 善用观察期：\n- 允许孩子在集体创作时先观察15-20分钟\n- 事后通过提问帮助反思（\"你注意到其他小朋友怎么玩的了吗？\"）\n\n4. 转换创造力表现形式：\n- 如果孩子不愿画画，可以改用语言创作故事\n- 通过整理分类游戏（如把玩具按新规则分组）培养创新思维\n\n记住，混合型孩子的创造力往往表现在意想不到的地方。建议记录孩子一周内的\"非常规解决方案\"，您可能会惊讶地发现：那些看似批判性的分析过程中，其实包含着独特的创造性火花。\n</answer>"
}

def format_prompt(example):
    # 和你的主脚本保持一致
    system_prompt = """
        你是一个专业且富有共情心的育婴专家，你的任务是回答家长的育婴提问，你需要根据婴幼儿的气质进行分析，给出合理的分析过程并给出答案。
        你需要首先思考家长的提问，用自己的话重新表述家长正面临什么样的育婴问题。
        接着你需要分析婴儿的气质特征，并分析婴儿当前的行为体现了什么气质特征，提醒自己可以让家长不用担心，安抚一下，结果放在 <temperment></temperment> 标签之间。
        然后你需要分析婴儿当前气质特征对应的养育策略，说明我应该怎么样做，避免和婴儿自身性格冲突的做法。分析放在 <strategy></strategy> 标签之间。
        最后给出养育建议，放在 <answer></answer> 标签之间。
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    # 我们只返回文本，以便后续观察
    return tokenizer.apply_chat_template(messages, tokenize=False)

# ========== 3. 应用模板并观察输出 ==========
formatted_text = format_prompt(one_shot)
print("\n" + "="*50)
print("【第一步】查看 apply_chat_template 生成的文本格式:")
print(formatted_text)
print("="*50 + "\n")


# ========== 4. 定义修正后的 tokenize 函数 ==========
# 这是验证的关键
def robust_tokenize_fn(text_example):
    # 首先，获取 Qwen 模板中关键 token 的 ID
    # 这是比硬编码字符串更可靠的方法
    im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # 获取 'assistant' 这个词本身的 token ID
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    
    # 按照你的主脚本进行 tokenize
    tokens = tokenizer(
        text_example,
        max_length=1024,
        truncation=True,
        # 这里我们先不 padding，方便观察真实长度
        # padding="max_length" 
    )
    
    input_ids = tokens["input_ids"]
    labels = list(input_ids) # 创建一个可修改的副本

    # 寻找 assistant 部分的起始点
    assistant_start_index = -1
    for i in range(len(input_ids) - 1):
        # 如果当前 token 是 <|im_start|> 并且下一个是 assistant
        if input_ids[i] == im_start_token_id and input_ids[i+1] == assistant_token_id:
            assistant_start_index = i
            break
            
    # 如果找到了 assistant 部分，就从头到该位置进行屏蔽
    if assistant_start_index != -1:
        for i in range(assistant_start_index):
            labels[i] = -100
    else:
        # 如果没找到，这是一个警告，可能数据格式有问题
        # 在这种情况下，我们保守地屏蔽所有标签，避免错误训练
        print("警告: 在样本中未找到 '<|im_start|>assistant' 结构！将屏蔽所有标签。")
        for i in range(len(labels)):
            labels[i] = -100

    tokens["labels"] = labels
    return tokens

# ========== 5. 执行 Tokenization 并验证 ==========
processed_data = robust_tokenize_fn(formatted_text)
input_ids = processed_data["input_ids"]
labels = processed_data["labels"]

print("\n" + "="*80)
print("【第二步】逐个 Token 检查 input_ids 和 labels 的对应关系:")
print(f"总 Token 数量: {len(input_ids)}")
print(f"{'Index':<6} | {'Token':<20} | {'Input ID':<10} | {'Label':<10}")
print("-"*80)

found_assistant_start = False
for i in range(len(input_ids)):
    # 为了视觉效果，当屏蔽结束，开始计算 loss 时，我们打印一个分割线
    if labels[i] != -100 and not found_assistant_start:
        print("\n" + "--- 屏蔽结束，从此开始计算 Loss ---".center(80) + "\n")
        found_assistant_start = True

    # 解码单个 token 来显示
    token_str = tokenizer.decode([input_ids[i]])
    
    print(f"{i:<6} | {token_str:<20} | {input_ids[i]:<10} | {labels[i]:<10}")
print("="*80 + "\n")

if found_assistant_start:
    print("✅ 验证成功！可以看到 'assistant' 之前的所有 'Label' 都被设置为了 -100。")
else:
    print("❌ 验证失败！未能正确找到 'assistant' 的起始位置。")