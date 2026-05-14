"""
衰减与空闲管理器
负责短期情绪(4D)向基线的指数回归，以及长时间未互动的空闲惩罚
"""

import asyncio
import time
import math
from typing import Optional, Dict

from astrbot.api import logger

from .emotion_utils import get_baseline, clamp


class DecayManager:
    def __init__(self, manager, config, unconscious_analyzer=None):
        self.manager = manager
        self.config = config
        self.unconscious = unconscious_analyzer
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())
            logger.info("[DecayManager] 后台平滑衰减任务已启动")

    async def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None
            logger.info("[DecayManager] 后台平滑衰减任务已停止")

    async def _run_loop(self):
        interval_minutes = self.config.get("decay_check_interval_minutes", 10)
        interval_seconds = interval_minutes * 60
        try:
            while True:
                await self._tick()
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            # 插件卸载或停止时优雅退出
            pass
        except Exception as e:
            logger.error(f"[DecayManager] 任务崩溃: {e}")

    async def _tick(self):
        """
        后台任务：同时处理个人情绪衰减与全局群聊氛围平复
        """
        if not self.config.get("decay_enabled", True):
            return
        
        now = time.time()
        
        # --- 第一部分：处理个人情绪衰减与空闲惩罚 ---
        all_keys = self.manager.get_all_keys()
        user_updates = {} # 批量更新容器
        
        for uid_key in all_keys:
            try:
                state = self.manager.user_data.get(uid_key)
                if not state:
                    continue
                
                last_interaction = state.get("last_interaction", now)
                last_update = state.get("last_update", now)
                elapsed_since_interaction = (now - last_interaction) / 3600.0
                dt_since_update = (now - last_update) / 3600.0
                
                user_patch = {}
                
                # 1.1 时间指数衰减 (向长期基线回归)
                decay_threshold = self.config.get("decay_threshold_hours", 3.0)
                if elapsed_since_interaction >= decay_threshold:
                    decay_patch = self._calc_exponential_decay(state, dt_since_update)
                    if decay_patch:
                        user_patch.update(decay_patch)
                
                # 1.2 空闲惩罚 (单峰函数判定)
                idle_threshold = self.config.get("idle_threshold_hours", 6.0)
                if (elapsed_since_interaction >= idle_threshold and 
                    not state.get("idle_penalty_applied", False)):
                    penalty_patch = await self._calc_idle_penalty(uid_key, state, elapsed_since_interaction)
                    if penalty_patch:
                        user_patch.update(penalty_patch)
                
                if user_patch:
                    user_patch["last_update"] = now
                    user_updates[uid_key] = user_patch
                    
            except Exception as e:
                logger.error(f"[DecayManager] 计算用户 {uid_key} 衰减失败: {e}")

        # 执行个人数据的批量保存
        if user_updates:
            await self.manager.batch_update(user_updates)


        # --- 第二部分：处理群聊/会话环境氛围 (Global Mood) 的极速衰减 ---
        # 环境情绪不向基线回归，而是向 0 (中立) 回归
        session_changed = False
        tau_global = self.config.get("global_time_decay_hours", 0.5) # 全局衰减常数
        
        # 遍历所有已记录的群聊/会话环境
        for sess_key, sess_state in self.manager.session_data.items():
            last_sess_update = sess_state.get("last_update", now)
            dt_sess = (now - last_sess_update) / 3600.0
            
            # 只有当环境偏移量不为 0 且流逝了一定时间(约3分钟)时才计算
            if dt_sess > 0.05:
                # 指数衰减因子
                factor = math.exp(-dt_sess / tau_global)
                
                mood_changed = False
                for dim_off in ["libido_offset", "aggression_offset", "self_libido_offset", "self_aggression_offset"]:
                    current_offset = sess_state.get(dim_off, 0.0)
                    
                    # 如果偏移量还没归零，就继续衰减
                    if abs(current_offset) > 0.01:
                        sess_state[dim_off] = current_offset * factor
                        mood_changed = True
                
                if mood_changed:
                    sess_state["last_update"] = now
                    session_changed = True

        # 如果环境氛围发生了变化，保存 session_data.json
        if session_changed:
            path = self.manager.data_path / "session_data.json"
            async with self.manager._lock:
                # 借用 manager 的同步保存逻辑
                await asyncio.to_thread(self.manager._save_sync_file, path, self.manager.session_data)
                
        if self.config.get("debug_mode") and (user_updates or session_changed):
            logger.debug(f"[DecayManager] 自动扫描完成。更新用户数: {len(user_updates)}, 环境氛围变动: {session_changed}")

    def _calc_exponential_decay(self, state: Dict, dt: float) -> Dict:
        """
        核心公式：new_val = base + (current - base) * exp(-dt / tau)
        dt: 距离上次更新的小时数
        """
        favour = state.get("favour", 0.0)
        baselines = get_baseline(favour) # (lib, agg, slib, sagg)
        
        # 衰减常数 tau。值越大回归越慢。默认 3.0 表示约3小时回归一大半。
        tau = self.config.get("decay_duration_hours", 3.0)
        # 衰减因子
        factor = math.exp(-dt / tau)
        
        patch = {}
        dims = ["libido", "aggression", "self_libido", "self_aggression"]
        
        for i, dim in enumerate(dims):
            current = state.get(dim, baselines[i])
            target_base = baselines[i]
            
            # 计算新值
            new_val = target_base + (current - target_base) * factor
            new_val = clamp(new_val, 0.0, 50.0)
            
            # 只有变化超过微小阈值才记录，避免无意义写入
            if abs(new_val - current) > 0.05:
                patch[dim] = new_val
        
        return patch

    async def _calc_idle_penalty(self, uid_key: str, state: Dict, elapsed_hours: float) -> Dict:
        """
        基于单峰函数的空闲惩罚计算
        曲线公式: P(t) = (t/12)^2 * exp(2 * (1 - t/12))
        
        特点：
        - 6小时开始产生明显感知。
        - 12小时达到不满情绪的顶峰 (Intensity = 1.0)。
        - 24小时后由于“情感脱敏”或“冷淡”，曲线开始下滑。
        - 结合持久化 user_status（如 sleeping, working）由 LLM 判定合理性权重。
        """
        import math
        
        # 1. 计算单峰时间曲线强度 P(t)
        # 以 12 小时为波峰中心
        t_norm = elapsed_hours / 12.0
        # 这一公式在 t=12 时结果为 1.0，t=24 时约为 0.54，t=48 时约为 0.04
        intensity_curve = (t_norm ** 2) * math.exp(2 * (1 - t_norm))
        
        # 2. 获取数据库中持久化的用户状态便签 (不再依赖聊天记录折叠)
        # 这个字段由潜意识 LLM 在对话时实时维护（如从 "active" 变为 "sleeping"）
        user_status = state.get("user_status", "active")
        
        # 3. 调用潜意识 LLM 获取失联合理性权重 W (0.0 ~ 1.0)
        llm_weight = 1.0 # 默认权重（完全无故失踪）
        llm_deltas = {}
        
        if self.unconscious:
            try:
                # 传入持久化的状态标签进行判定
                # 即使对话记录被硬折叠，state 里的状态标签依然物理存在
                res = await self.unconscious.analyze_idle(uid_key, elapsed_hours, user_status)
                if res:
                    llm_weight = float(res.get("weight", 1.0))
                    llm_deltas = res
            except Exception as e:
                logger.warning(f"[DecayManager] 潜意识空闲分析失败: {e}")

        # 4. 最终影响系数 = 曲线强度 P(t) * LLM合理性权重 W * 配置灵敏度
        sensitivity = self.config.get("idle_penalty_sensitivity", 0.3)
        final_factor = intensity_curve * llm_weight * sensitivity
        
        # 准备更新补丁
        patch = {"idle_penalty_applied": True}
        
        # 5. 定义基础惩罚倾向 (即 P(t)=1.0 且 W=1.0 时的最大增量)
        # 默认方向：防备心(Aggression)大幅上升，亲近欲(Libido)下降
        base_directions = {
            "aggression": 6.0,        # 峰值增加 6 点防备心
            "libido": -4.0,           # 峰值降低 4 点亲近欲
            "self_aggression": 2.0,   # 产生少许被抛弃的负罪感
            "self_libido": -1.0       # 轻微自信下降
        }
        
        for dim, base_val in base_directions.items():
            # 允许潜意识 LLM 修正增量方向（例如某些状态下失踪会让 AI 更担心而不是更生气）
            direction_val = float(llm_deltas.get(f"{dim}_delta", base_val))
            
            # 计算最终变化值并应用裁剪
            current_val = state.get(dim, 25.0)
            delta = direction_val * final_factor
            patch[dim] = clamp(current_val + delta, 0.0, 50.0)
            
        if self.config.get("debug_mode"):
            logger.debug(
                f"[IdlePenalty] 用户 {uid_key} | 时长: {elapsed_hours:.1f}h | 记录状态: {user_status}\n"
                f">> 曲线强度 P(t): {intensity_curve:.2f} | LLM 权重 W: {llm_weight:.2f}\n"
                f">> 最终影响系数: {final_factor:.4f}"
            )
            
        return patch