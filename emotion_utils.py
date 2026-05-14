"""
情绪描述工具模块
提供基线计算、情绪标签生成、数值裁剪等纯函数
"""

from typing import Tuple

def get_baseline(favour: float) -> Tuple[float, float, float, float]:
    """
    根据长期好感度(favour)计算各维度的目标基线值
    
    Args:
        favour: 好感度，范围 -100 ~ 100
    
    Returns:
        (libido_base, aggression_base, self_libido_base, self_aggression_base)
        均限定在 0~50 范围内
    """
    # 将 favour -100..100 映射到 0..1
    t = (favour + 100) / 200.0
    t = max(0.0, min(1.0, t))
    
    libido_base = 10.0 + t * 30.0      # 10 ~ 40
    aggression_base = 30.0 - t * 25.0  # 30 ~ 5
    self_libido_base = 15.0 + t * 20.0 # 15 ~ 35
    self_aggression_base = 20.0 - t * 15.0  # 20 ~ 5
    
    # 确保边界
    return (libido_base, aggression_base, self_libido_base, self_aggression_base)


def clamp(value: float, min_val: float = 0.0, max_val: float = 50.0) -> float:
    """数值裁剪"""
    return max(min_val, min(max_val, value))


def get_emotion_description(state: dict) -> str:
    """
    根据长期记忆与短期情绪生成自然语言描述，用于注入主LLM
    
    Args:
        state: 包含 favour, attitude, relationship, libido, aggression 等字典
    
    Returns:
        一段综合的自然语言描述
    """
    favour = state.get('favour', 0.0)
    attitude = state.get('attitude', '中立')
    relationship = state.get('relationship', '陌生人')
    
    libido = state.get('libido', 25.0)
    aggression = state.get('aggression', 25.0)
    self_libido = state.get('self_libido', 25.0)
    self_aggression = state.get('self_aggression', 25.0)
    
    # ----- 对他短期情绪描述 -----
    # 力比多描述
    if libido >= 35:
        toward_libido = "渴望亲近，黏人"
    elif libido >= 20:
        toward_libido = "愿意接近，温和"
    else:
        toward_libido = "保持距离，冷淡"
    
    # 攻击性描述
    if aggression >= 35:
        toward_aggression = "带有明显敌意"
    elif aggression >= 20:
        toward_aggression = "略有不耐烦"
    else:
        toward_aggression = "宽容温顺"
    
    # 特殊组合处理：高好感+高攻击性 → 撒娇式吃醋
    if favour >= 30 and aggression >= 35:
        toward_aggression = "撒娇式吃醋"
    elif favour <= -30 and aggression >= 35:
        toward_aggression = "充满防备与敌意"
    
    toward_text = f"{toward_libido}，{toward_aggression}"
    
    # ----- 自身短期状态描述 -----
    # 自力比多
    if self_libido >= 35:
        self_text = "自信自爱"
    elif self_libido >= 20:
        self_text = "心态平和"
    else:
        self_text = "自我怀疑"
    
    # 自攻击性
    if self_aggression >= 35:
        self_text += "，充满自责"
    elif self_aggression >= 20:
        self_text += "，略显愧疚"
    else:
        self_text += "，接纳自己"
    
    # 特殊悲剧状态：高自攻击性 + 低自力比多 → 自毁倾向
    if self_aggression >= 37.5 and self_libido <= 12.5:
        self_text = "陷入自我否定，情绪崩溃边缘"
        
    # 组合长期记忆（慢变量）与短期情绪（快变量）
    return (
        f"【长期关系】：{relationship} (好感：{int(favour)})；"
        f"【总体印象】：{attitude}。\n"
        f"【当下情绪波动】-> 对用户：{toward_text}；自身状态：{self_text}。"
    )


# emotion_utils.py

def get_fuzzy_state_report(state: dict) -> str:
    """
    生成模糊化的状态报告，用于管理员快捷查询。
    包含：好感等级标签、长期记忆描述、情绪自然语言概括。
    """
    favour = state.get('favour', 0.0)
    attitude = state.get('attitude', '中立')
    relationship = state.get('relationship', '陌生人')
    
    # 1. 好感度区间标签（8个区间，每个 25 分）
    favour_labels = [
        "极度敌视", # -100 ~ -75
        "厌恶反感", # -75 ~ -50
        "警惕疏远", # -50 ~ -25
        "略带隔阂", # -25 ~ 0
        "萍水相逢", # 0 ~ 25
        "友好信任", # 25 ~ 50
        "亲密依赖", # 50 ~ 75
        "至死不渝"  # 75 ~ 100
    ]
    # 计算索引：将 -100~100 映射到 0~7
    idx = int((favour + 100) / 25)
    idx = max(0, min(7, idx))
    favour_tag = favour_labels[idx]

    # 2. 复用情绪描述逻辑（快变量）
    libido = state.get('libido', 25.0)
    aggression = state.get('aggression', 25.0)
    self_libido = state.get('self_libido', 25.0)
    self_aggression = state.get('self_aggression', 25.0)
    
    # 对他部分
    if libido >= 35:
        t_lib = "渴望亲近"
    elif libido >= 20:
        t_lib = "愿意接触"
    else:
        t_lib = "保持距离"

    if aggression >= 35:
        if favour >= 30: t_agg = "撒娇吃醋"
        elif favour <= -30: t_agg = "充满敌意"
        else: t_agg = "非常急躁"
    elif aggression >= 20:
        t_agg = "略显不耐"
    else:
        t_agg = "宽容温顺"

    # 自身部分
    if self_aggression >= 37.5 and self_libido <= 12.5:
        s_text = "自我否定，几近崩溃"
    else:
        s_lib = "自信" if self_libido >= 35 else ("平和" if self_libido >= 20 else "自卑")
        s_agg = "自责" if self_aggression >= 35 else ("愧疚" if self_aggression >= 20 else "接纳")
        s_text = f"{s_lib}且{s_agg}"

    # 3. 组装报告
    report = (
        f"【好感评价】：{favour_tag}\n"
        f"【当前关系】：{relationship}\n"
        f"【核心印象】：{attitude}\n"
        f"【对他状态】：{t_lib}，{t_agg}\n"
        f"【内心状态】：{s_text}"
    )
    return report