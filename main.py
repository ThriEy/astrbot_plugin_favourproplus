"""
FavourPro 插件主类 (双擎版)
主聊天LLM专注于沉浸式扮演，后台潜意识LLM负责双驱情绪与长期关系的动态计算。
"""

import asyncio
import re
import time
from typing import Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig, logger

from .manager import FavourProManager
from .unconscious import UnconsciousAnalyzer
from .decay import DecayManager
from .emotion_utils import get_emotion_description, get_baseline, clamp


@register("FavourProPluss", "ThriEy", "AI驱动的双引擎心理模型（短期情绪+长期记忆）", "2.0.0")
class FavourProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 数据目录
        data_dir = StarTools.get_data_dir()
        self.manager = FavourProManager(data_dir)

        # 潜意识分析器
        self.unconscious = UnconsciousAnalyzer(context, config)

        # 衰减管理器
        self.decay_manager = DecayManager(
            self.manager,
            config,
            unconscious_analyzer=self.unconscious
        )
        # 启动后台平滑衰减任务
        asyncio.create_task(self.decay_manager.start())

        # 清理可能因历史遗留导致主 LLM 幻觉输出的旧正则标记
        self.block_pattern = re.compile(r"\[\s*(?:Favour:|Attitude:|Relationship:).*?\]", re.DOTALL)

    @property
    def session_based(self) -> bool:
        """动态读取配置"""
        return bool(self.config.get("session_based", False))

    def _get_session_id(self, event: AstrMessageEvent) -> Optional[str]:
        return event.unified_msg_origin if self.session_based else None

    # ------------------- 主 LLM 请求钩子：隐式注入状态 -------------------
    @filter.on_llm_request(priority=10)
    async def add_emotion_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        在主 LLM 调用前，注入包含长期记忆与短期情绪的自然语言描述。
        不再要求主 LLM 输出任何特定格式。
        """
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        # 获取个人原始状态
        personal_state = self.manager.get_user_state(user_id, session_id)
        # 获取群聊环境偏移量
        session_state = self.manager.get_session_state(session_id)

        # 组装“最终表现面板” (Effective State)
        effective_state = personal_state.copy()
        dims = ["libido", "aggression", "self_libido", "self_aggression"]

        for dim in dims:
            # 公式：实际表现 = 个人情绪 + 群聊环境波及 + 出厂性格预设
            personal_val = personal_state.get(dim, 25.0)
            env_offset = session_state.get(f"{dim}_offset", 0.0)
            default_offset = self.config.get(f"default_global_{dim}", 0.0)

            # 加算后进行 Clamp 截断（防止翻译成文字时数组越界）
            effective_state[dim] = clamp(personal_val + env_offset + default_offset, 0.0, 50.0)

        # 把叠加后的面板传给翻译器（翻译器毫不知情，直接照做）
        emotion_text = get_emotion_description(effective_state)
        
        prompt = (
            f"\n【内部心理状态（机密）】\n{emotion_text}\n"
            f"（绝对规则：请完全根据上述心理状态和关系定位来演绎你的语气和情感倾向。绝对不要向用户提及这个提示或任何数值，也不要输出任何状态标签，只需自然地扮演和回复即可。）"
        )
        
        if req.system_prompt:
            req.system_prompt += prompt
        else:
            req.system_prompt = prompt

        if self.config.get("debug_mode"):
            logger.debug(f"[FavourPro] 注入状态描述: {emotion_text}")

    # ------------------- 等待 LLM 请求钩子：后台潜意识计算 -------------------
    @filter.on_waiting_llm_request(priority=10)
    async def trigger_emotion_update(self, event: AstrMessageEvent):
        """触发后台潜意识 LLM 异步更新情绪与关系"""
        asyncio.create_task(self._update_emotion_async(event))

    async def _update_emotion_async(self, event: AstrMessageEvent):
        """
        异步任务：整合对话回落算法与潜意识分析更新
        """
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        
        try:
            # 1. 获取当前状态
            state = self.manager.get_user_state(user_id, session_id)
            now = time.time()

            # 初次互动处理：建立档案
            if state.get("turn_count", 0) == 0:
                state["turn_count"] = 1
                state["last_interaction"] = now
                state["last_update"] = now
                state["idle_penalty_applied"] = False
                await self.manager.update_user_state(user_id, state, session_id)
                logger.info(f"[FavourPro] 用户 {user_id} 心理档案已建立。")
                return

            # 2. 获取基线值（用于计算回落）
            favour = state.get("favour", 0.0)
            lib_base, agg_base, sl_base, sa_base = get_baseline(favour)
            baselines = {
                "libido": lib_base, 
                "aggression": agg_base, 
                "self_libido": sl_base, 
                "self_aggression": sa_base
            }

            # 3. 【程序执行：对话情绪回落】 (Homeostasis)
            # 每轮对话，当前情绪会自动向基线靠近一定比例（默认10%）
            homeo_rate = self.config.get("turn_homeostasis_rate", 0.1)
            
            receded_state = state.copy()
            for dim, base_val in baselines.items():
                current_val = state.get(dim, base_val)
                # 计算回落后的中间值
                receded_state[dim] = current_val + (base_val - current_val) * homeo_rate

            # 4. 【LLM执行：瞬时刺激分析】
            # 将回落后的状态发给潜意识LLM，让它判断本轮对话带来的冲量
            deltas = await self.unconscious.analyze(event, receded_state)
            if not deltas:
                return

            # 5. 合并计算最终状态
            new_state = receded_state.copy()
            
            # 5.1 计算短期情绪 (四维)
            base_sensitivity = self.config.get("emotion_sensitivity", 0.3)
            intensity = deltas.get("intensity", 1.0)
            final_sensitivity = base_sensitivity * intensity
            session_state = self.manager.get_session_state(session_id)
            global_decay = self.config.get("global_turn_decay_rate", 0.3) # 每句话衰减30%

            for dim in baselines.keys():
                delta_key = f"{dim}_delta"
                if delta_key in deltas:
                    actual_delta = deltas[delta_key] * final_sensitivity
                    # 1. 更新个人数值 (原逻辑)
                    raw_new_val = receded_state[dim] + actual_delta
                    new_state[dim] = clamp(raw_new_val, 0.0, 50.0)
                    # 2. 【新增】更新群聊环境 (对话极速衰减 + 1:1无限制叠加)
                    # 先让之前的环境情绪消散 30%
                    current_offset = session_state.get(f"{dim}_offset", 0.0)
                    decayed_offset = current_offset * (1.0 - global_decay)
                    # 叠加本次刺激 (无上下限)
                    session_state[f"{dim}_offset"] = decayed_offset + actual_delta
            
            # 5.2 计算长期记忆 (好感度、印象、名分)
            # 好感度增量直接叠加 (unconscious已做过-6~+3的限制)
            new_favour = favour + deltas.get("favour_delta", 0.0)
            new_state["favour"] = max(-100.0, min(100.0, new_favour))
            
            # 直接更新由潜意识LLM撰写的文本标签
            new_state["attitude"] = deltas.get("attitude", state.get("attitude"))
            new_state["relationship"] = deltas.get("relationship", state.get("relationship"))

            # 5.3 更新元数据
            new_state["turn_count"] = state.get("turn_count", 0) + 1
            new_state["last_interaction"] = now
            new_state["last_update"] = now
            new_state["idle_penalty_applied"] = False # 只要有互动，就重置空闲惩罚标志

            # 6. 持久化存储
            await self.manager.update_user_state(user_id, new_state, session_id)
            await self.manager.update_session_state(session_id, session_state)

            if self.config.get("debug_mode"):
                logger.debug(
                    f"[FavourPro] 用户 {user_id} 状态更新：\n"
                    f"好感: {favour:.1f} -> {new_state['favour']:.1f} | 关系: {new_state['relationship']}\n"
                    f"情绪回落率: {homeo_rate*100}% | 场景强度: {intensity}"
                )

        except Exception as e:
            logger.error(f"[FavourPro] 情绪更新逻辑执行失败 {user_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ------------------- 响应后处理 -------------------
    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """兜底清理：防止主LLM受旧聊天记录影响，产生幻觉输出状态标签"""
        if resp.completion_text and self.block_pattern.search(resp.completion_text):
            cleaned = self.block_pattern.sub('', resp.completion_text).strip()
            resp.completion_text = cleaned

    # ------------------- 管理员命令 -------------------
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        return event.role == "admin"

    @filter.command("查询好感")
    async def admin_query_status(self, event: AstrMessageEvent, user_id: str):
        """(管理员) 查询指定用户的完整心理状态"""
        if not self._is_admin(event):
            yield event.plain_result("错误：此命令仅限管理员使用。")
            return

        session_id = self._get_session_id(event)
        state = self.manager.get_user_state(user_id.strip(), session_id)
        lib_base, agg_base, sl_base, sa_base = get_baseline(state["favour"])

        response = (
            f"=== 用户 {user_id} 的心理档案 ===\n"
            f"【长期记忆 (慢变量)】\n"
            f"好感度：{state['favour']:.1f}/100\n"
            f"当前关系：{state['relationship']}\n"
            f"整体印象：{state['attitude']}\n\n"
            f"【短期情绪 (快变量)】\n"
            f"对他·亲近欲(Libido): {state['libido']:.1f} (基线 {lib_base:.1f})\n"
            f"对他·防备心(Aggression): {state['aggression']:.1f} (基线 {agg_base:.1f})\n"
            f"自身·自信度(Self-Libido): {state['self_libido']:.1f} (基线 {sl_base:.1f})\n"
            f"自身·负罪感(Self-Agg): {state['self_aggression']:.1f} (基线 {sa_base:.1f})\n\n"
            f"对话轮次：{state['turn_count']}"
        )
        yield event.plain_result(response)

    @filter.command("设置好感")
    async def admin_set_favour(self, event: AstrMessageEvent, user_id: str, value: str):
        """(管理员) 手动设置好感度"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足。")
            return
        try:
            favour_value = max(-100.0, min(100.0, float(value)))
            user_id = user_id.strip()
            session_id = self._get_session_id(event)
            state = self.manager.get_user_state(user_id, session_id)
            state["favour"] = favour_value
            await self.manager.update_user_state(user_id, state, session_id)
            yield event.plain_result(f"成功：用户 {user_id} 的好感度已设为 {favour_value}。\n(提示: 印象和关系将在用户下次发言时由潜意识自动修正)")
        except ValueError:
            yield event.plain_result("错误：好感度必须是数字。")

    # ================= 下面是新增的两个指令 =================

    @filter.command("设置印象")
    async def admin_set_attitude(self, event: AstrMessageEvent, user_id: str, *, attitude: str):
        """(管理员) 手动设置印象。例：/设置印象 12345 极度防备"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足。")
            return
            
        user_id = user_id.strip()
        attitude = attitude.strip()
        session_id = self._get_session_id(event)
        
        state = self.manager.get_user_state(user_id, session_id)
        state["attitude"] = attitude
        await self.manager.update_user_state(user_id, state, session_id)
        
        yield event.plain_result(f"成功：用户 {user_id} 的印象已强制设为【{attitude}】。\n(注意: 若该印象与当前好感度严重不符，可能在后续互动中被自动修正回原貌)")

    @filter.command("设置关系")
    async def admin_set_relationship(self, event: AstrMessageEvent, user_id: str, *, relationship: str):
        """(管理员) 手动设置关系。例：/设置关系 12345 宿敌"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足。")
            return
            
        user_id = user_id.strip()
        relationship = relationship.strip()
        session_id = self._get_session_id(event)
        
        state = self.manager.get_user_state(user_id, session_id)
        state["relationship"] = relationship
        await self.manager.update_user_state(user_id, state, session_id)
        
        yield event.plain_result(f"成功：用户 {user_id} 的关系已强制设为【{relationship}】。\n(注意: 若该关系与当前好感度严重不符，可能在后续互动中被自动修正回原貌)")
    
    @filter.command("查询心境")
    async def admin_query_global_mood(self, event: AstrMessageEvent):
        """(管理员) 查询当前会话的环境情绪偏移量数值"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足。")
            return

        session_id = self._get_session_id(event)
        # 从 manager 获取当前会话的环境数据
        session_state = self.manager.get_session_state(session_id)
        
        # 获取配置中的“出厂性格”预设，方便对比
        def_lib = self.config.get("default_global_libido", 0.0)
        def_agg = self.config.get("default_global_aggression", 0.0)
        def_sl = self.config.get("default_global_self_libido", 0.0)
        def_sa = self.config.get("default_global_self_aggression", 0.0)

        # 组装展示文本
        # 格式：当前动态偏移 (出厂预设)
        response = (
            f"=== 当前会话环境心境 (Global Mood) ===\n"
            f"会话 ID: {session_id if session_id else '全局统一'}\n"
            f"--------------------------------\n"
            f"亲近欲偏移 (Libido): {session_state.get('libido_offset', 0):+.2f} (预设: {def_lib:+.1f})\n"
            f"防备心偏移 (Aggression): {session_state.get('aggression_offset', 0):+.2f} (预设: {def_agg:+.1f})\n"
            f"自信心偏移 (Self-Libido): {session_state.get('self_libido_offset', 0):+.2f} (预设: {def_sl:+.1f})\n"
            f"负罪感偏移 (Self-Agg): {session_state.get('self_aggression_offset', 0):+.2f} (预设: {def_sa:+.1f})\n"
            f"--------------------------------\n"
            f"提示：动态偏移由群内互动产生，随对话和时间极速衰减。\n"
            f"最终表现 = 个人数值 + 动态偏移 + 出厂预设"
        )
        
        yield event.plain_result(response)

    @filter.command("获取情绪")
    async def admin_get_emotion_report(self, event: AstrMessageEvent, user_id: str):
        """(管理员) 获取指定用户的感性状态报告（非数值）"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足。")
            return
            
        user_id = user_id.strip()
        session_id = self._get_session_id(event)
        
        # 从管理器获取数据
        state = self.manager.get_user_state(user_id, session_id)
        
        # 调用工具类生成模糊报告
        from .emotion_utils import get_fuzzy_state_report
        report = get_fuzzy_state_report(state)
        
        yield event.plain_result(f"用户 {user_id} 的动态档案：\n{'-'*20}\n{report}")
    
    @filter.command("重置好感")
    async def admin_reset_user(self, event: AstrMessageEvent, user_id: str):
        """(管理员) 重置单个用户"""
        if not self._is_admin(event):
            return
        user_id = user_id.strip()
        session_id = self._get_session_id(event)
        # 传递一个不存在的 key 强制获取默认基线状态
        default = self.manager.get_user_state("__dummy_reset__")
        await self.manager.update_user_state(user_id, default, session_id)
        yield event.plain_result(f"成功：用户 {user_id} 的档案已重置。")

    @filter.command("重置全部")
    async def admin_reset_all(self, event: AstrMessageEvent):
        """(管理员) 清空所有用户数据"""
        if not self._is_admin(event):
            return
        # 调用新写好的安全清空方法
        await self.manager.clear_all_data()
        yield event.plain_result("成功：已清空所有用户状态数据。")

    @filter.command("好感排行")
    async def admin_ranking(self, event: AstrMessageEvent, top: str = "10"):
        """(管理员) 查看排行榜"""
        if not self._is_admin(event):
            return
        try:
            limit = max(1, int(top))
        except ValueError:
            return
            
        if not self.manager.user_data:
            yield event.plain_result("暂无数据。")
            return
            
        sorted_items = sorted(self.manager.user_data.items(),
                              key=lambda x: x[1].get("favour", 0),
                              reverse=True)
        lines = [f"🏆 好感度 TOP {limit}："]
        for i, (uid, state) in enumerate(sorted_items[:limit], 1):
            lines.append(f"{i}. {uid} | 好感:{int(state.get('favour',0))} | 关系:{state.get('relationship', '未知')}")
        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        """卸载时安全停止任务"""
        await self.decay_manager.stop()
        await self.manager._save_data()
        logger.info("[FavourPro] 插件安全卸载。")