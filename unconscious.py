"""
潜意识分析模块
异步调用轻量级LLM分析用户消息，返回短期情绪增量与长期关系状态
"""

import json
import re
from typing import Dict, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.conversation_mgr import Conversation

from .emotion_utils import get_baseline


class UnconsciousAnalyzer:
    def __init__(self, context, config):
        self.context = context
        self.config = config

    async def analyze(self, event: AstrMessageEvent, user_state: Dict) -> Dict:
        """
        分析用户消息，返回情绪增量及长期关系更新
        """
        # 获取对话历史（最近5条左右）
        conv_mgr = self.context.conversation_manager
        umo = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(umo)
        conversation: Conversation = await conv_mgr.get_conversation(umo, curr_cid)
        history_text = conversation.history if conversation else ""
        # 截取最近2000字符
        history_snippet = history_text[-2000:] if len(history_text) > 2000 else history_text

        prompt = self._build_prompt(
            user_state,
            history_snippet,
            event.message_str,
            user_state.get("turn_count", 0)
        )

        llm_config = self.config.get("unconscious_llm", {})
        provider_id = llm_config.get("provider_id")
        if not provider_id:
            # 若无专用provider，回退到当前默认聊天provider
            provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        if not provider_id:
            logger.warning("[Unconscious] 无可用 LLM provider，跳过分析")
            return self._default_deltas(user_state)

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                contexts=[],
                system_prompt="你是一个后台潜意识分析器，负责评估用户的短期情绪波动与长期关系沉淀。只输出严格的JSON。",
            )
            text = resp.completion_text
            logger.debug(f"[Unconscious] LLM 返回: {text[:200]}")
            
            deltas = self._parse_json(text, user_state)
            deltas = self._clamp_deltas(deltas)
            return deltas
        except Exception as e:
            logger.error(f"[Unconscious] LLM 调用失败: {e}")
            return self._default_deltas(user_state)

    async def analyze_idle(self, uid: str, elapsed_hours: float, user_status: str) -> Optional[Dict]:
        """
        空闲惩罚分析：结合持久化状态标签判定失联合理性
        """
        prompt = self._build_idle_prompt(elapsed_hours, user_status)
        llm_config = self.config.get("unconscious_llm", {})
        provider_id = llm_config.get("provider_id") or await self.context.get_current_chat_provider_id(uid)
        
        if not provider_id:
            return None
            
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                contexts=[],
                system_prompt="你是一个情绪分析器，负责评估用户行为的合理性。只输出严格的 JSON。",
            )
            # 调用已有的 _parse_json
            return self._parse_json(resp.completion_text, {})
        except Exception as e:
            logger.error(f"[Unconscious] 空闲分析 LLM 调用失败: {e}")
            return None

    def _build_prompt(self, user_state: Dict, history: str, latest_msg: str, turn_count: int) -> str:
        favour = user_state.get("favour", 0.0)
        attitude = user_state.get("attitude", "中立")
        relationship = user_state.get("relationship", "陌生人")
        status = user_state.get("user_status", "活跃")
        libido = user_state.get("libido", 25.0)
        aggression = user_state.get("aggression", 25.0)
        self_libido = user_state.get("self_libido", 25.0)
        self_aggression = user_state.get("self_aggression", 25.0)
        
        libido_base, agg_base, self_lib_base, self_agg_base = get_baseline(favour)
        
        return f"""作为潜意识分析器，请根据对话历史，输出本次互动引起的状态变化（JSON格式）。

**一、 短期情绪规则 (快变量)**：
所有四个维度的波动范围均为 -2.0 到 2.0，且必须根据用户消息内容给出有意义的、微小的变化。

1. 他力比多（libido_delta）与他攻击性（aggression_delta）：
   - 用户表达善意、关心、亲近 → libido_delta 为正，aggression_delta 为负。
   - 用户表达恶意、冷漠、疏远 → libido_delta 为负，aggression_delta 为正。

2. 自力比多（self_libido_delta）与自攻击性（self_aggression_delta）：
   - 用户给予肯定、鼓励、赞美 → self_libido_delta 为正，self_aggression_delta 为负或零。
   - 用户贬低、嘲讽、忽视 → self_libido_delta 为负，同时 self_aggression_delta 为正。
   - 用户表达自身脆弱或道歉 → self_aggression_delta 通常为负（减轻心理负担），但若对方态度过于卑微可能引发轻微反向波动（视上下文而定）。
   - 在对话氛围平淡、仅为信息交换时，自情绪变化应趋于零。

3. 场景强度（intensity）：
   - 日常闲聊 → 1.0
   - 包含明显情感色彩（争执、安慰、表白等） → 1.5 ~ 2.0
   - 极端冲突或情感爆发 → 2.0

**二、 长期关系规则 (慢变量)**：
1. 长期好感度(favour)的调整必须严格遵循非对称原则：**增加极难且缓慢（一般+0 到 +1），下降容易且显著（-1 到 -6）**。绝大多数普通对话 favour_delta 应为 0。
2. 你需要亲自撰写最新的 `attitude` (你对他的印象) 和 `relationship` (你们的关系定位)，必须与调整后的好感度完全吻合！
   - [91~100 亲密依靠]: 挚爱、极度依赖。
   - [70~90 亲密信赖]: 恋人/挚友、充满信任。
   - [30~69 友好]: 朋友/熟人、乐于协助。
   - [-20~29 中立礼貌]: 陌生人/过客、保持距离。
   - [-60~-11 反感]: 讨厌的人、冷淡不耐烦。
   - [-100~-61 厌恶敌对]: 死敌、极其抵触。

   **三、 状态追踪 **:
根据对话判断用户目前的处境，并更新 `user_status` 字段。
- 用简单的短语概况用户目前的状态，如“在玩游戏，准备去睡觉”
- 如果用户对话中表达了状态（吃饭、打游戏、工作等），或者表达了意图（如去睡觉、准备洗澡、去开会、准备出门），将其更新。
- 如果对话没提到状态改变或者意图表达，请保持原样。

**当前状态**：
- 长期好感度: {favour:.1f}/100
- 当前关系: {relationship}
- 当前印象: {attitude}
- 用户状态：{status}
- 短期他力比多(亲近欲): {libido:.1f}/50 (基线{libido_base:.1f})
- 短期他攻击性(防备心): {aggression:.1f}/50 (基线{agg_base:.1f})
- 短期自力比多(自信): {self_libido:.1f}/50
- 短期自攻击性(自责): {self_aggression:.1f}/50

最近对话历史：
{history}

用户最新消息：{latest_msg}

输出格式（严格返回JSON，不要Markdown代码块，不要其他文字）：
{{
  "libido_delta": 0.0,
  "aggression_delta": 0.0,
  "self_libido_delta": 0.0,
  "self_aggression_delta": 0.0,
  "favour_delta": 0.0,
  "intensity": 1.0,
  "attitude": "简短的一句话印象",
  "relationship": "关系定位词",
  "user_status":"睡觉"
}}"""

    def _build_idle_prompt(self, elapsed_hours: float,user_status: str) -> str:
        return f"""用户已经 {elapsed_hours:.1f} 小时没有互动了。
请分析这次“失联”的合理性，并给出情绪影响权重 (0.0 ~ 1.0)。
【前情提要】
- 用户最后离开时，系统记录的活动状态标签为：[{user_status}]
- 用户已经连续失联此时长：{elapsed_hours:.1f} 小时
**分析维度**：
1. 时间段：若是深夜（如23:00-07:00）属于正常休息，权重应趋近 0。
2. 时长：当前消失时长为 {elapsed_hours:.1f} 小时。
3. 状态对齐：若上次状态很愉快，突然消失可能导致更高失落感；若上次在吵架，消失可能被视为逃避。
   检查失联时长是否符合离开时的状态。
   - 若状态为“去睡觉”，失联 6-9 小时是极度合理的（W 趋近 0）。
   - 若状态为“忙碌/开会”，失联 1-4 小时是合理的（W 较低）。
   - 若状态为“活跃/玩游戏”却突然消失 6-12 小时，属于无故失踪（W 较高）。
**权重 W 定义**：
- 0.0：完全理解，没有任何负面情绪。
- 1.0：完全无法理解，产生最大的被冷落感、愤怒或失落。
- 权重 W 和情绪变化基础倾向是相互独立的。权重变化衡量心理上的合理性，情绪变化基础倾向衡量造成的情绪冲击

只返回 JSON：
{{
  "reasoning": "简短的理由分析",
  "weight": 0.5, 
  "libido_delta": -1.0, // 情绪变化基础倾向(范围-4到4)
  "aggression_delta": 1.0, 
  "self_aggression_delta": 0.5
}}"""

    def _parse_json(self, text: str, current_state: Dict) -> dict:
        default = self._default_deltas(current_state)
        try:
            # 清理可能的 markdown 标记
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            
            # 提取字段，如果不存在则使用默认值（保留原有字符串）
            parsed = default.copy()
            for key in ["libido_delta", "aggression_delta", "self_libido_delta", "self_aggression_delta", "favour_delta", "intensity","weight"]:
                if key in data:
                    parsed[key] = data[key]
                    
            if "attitude" in data and isinstance(data["attitude"], str):
                parsed["attitude"] = data["attitude"]
            if "relationship" in data and isinstance(data["relationship"], str):
                parsed["relationship"] = data["relationship"]
                
            return parsed
        except Exception:
            # 正则 fallback
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    parsed = default.copy()
                    for key in ["libido_delta", "aggression_delta", "self_libido_delta", "self_aggression_delta", "favour_delta"]:
                        if key in data: parsed[key] = data[key]
                    if "attitude" in data: parsed["attitude"] = str(data["attitude"])
                    if "relationship" in data: parsed["relationship"] = str(data["relationship"])
                    return parsed
                except Exception:
                    pass
        logger.warning(f"[Unconscious] JSON解析失败，使用默认值。原始返回: {text[:100]}...")
        return default

    def _clamp_deltas(self, data: dict) -> dict:
        clamped = data.copy()
        # 短期情绪波动限制在 -2.0 到 2.0 之间
        clamped["libido_delta"] = max(-2.0, min(2.0, float(data.get("libido_delta", 0.0))))
        clamped["aggression_delta"] = max(-2.0, min(2.0, float(data.get("aggression_delta", 0.0))))
        clamped["self_libido_delta"] = max(-2.0, min(2.0, float(data.get("self_libido_delta", 0.0))))
        clamped["self_aggression_delta"] = max(-2.0, min(2.0, float(data.get("self_aggression_delta", 0.0))))
        
        # 长期好感度波动限制在 -6.0 到 2.0 之间
        clamped["favour_delta"] = max(-6.0, min(2.0, float(data.get("favour_delta", 0.0))))
        
        intensity = float(data.get("intensity", 1.0))
        clamped["intensity"] = max(0.5, min(4.0, intensity))
        
        return clamped

    def _default_deltas(self, state: Dict):
        return {
            "libido_delta": 0.0,
            "aggression_delta": 0.0,
            "self_libido_delta": 0.0,
            "self_aggression_delta": 0.0,
            "favour_delta": 0.0,
            "intensity": 1.0,
            "attitude": state.get("attitude", "中立"),
            "relationship": state.get("relationship", "陌生人"),
        }