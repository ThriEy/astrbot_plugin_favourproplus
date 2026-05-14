"""
数据管理模块
持有用户状态，提供 CRUD 操作，支持异步锁和字段迁移
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

from astrbot.api import logger

from .emotion_utils import get_baseline, clamp


class FavourProManager:
    DEFAULT_ATTITUDE = "中立"
    DEFAULT_RELATIONSHIP = "陌生人"
    DEFAULT_STATUS = "活跃"

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._lock = asyncio.Lock()
        self._init_path()
        self.user_data = self._load_data("user_data.json")
        self.session_data = self._load_data("session_data.json") # 【新增】加载群聊环境数据
        self._migrate_all_users()

    def _init_path(self):
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _load_data(self, filename: str) -> Dict[str, Any]:
        path = self.data_path / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[FavourPro] 加载数据失败 {path}: {e}")
            return {}

    async def _save_data(self):
        """独立的保存方法（供退出时使用）"""
        async with self._lock:
            path = self.data_path / "user_data.json"
            await asyncio.to_thread(self._save_sync, path)

    def _save_sync(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.user_data, f, ensure_ascii=False, indent=2)

    def _migrate_all_users(self):
        changed = False
        for key, state in self.user_data.items():
            if self._ensure_full_fields(state):
                changed = True
        if changed:
            self._save_sync(self.data_path / "user_data.json")

    def _ensure_full_fields(self, state: dict) -> bool:
        need_save = False
        if "favour" not in state:
            state["favour"] = 0.0
            need_save = True
        if "attitude" not in state:
            state["attitude"] = self.DEFAULT_ATTITUDE
            need_save = True
        if "relationship" not in state:
            state["relationship"] = self.DEFAULT_RELATIONSHIP
            need_save = True
        
        favour = state["favour"]
        lib_b, agg_b, slib_b, sagg_b = get_baseline(favour)
        
        field_map = {
            "libido": lib_b, "aggression": agg_b,
            "self_libido": slib_b, "self_aggression": sagg_b
        }
        for field, base_val in field_map.items():
            if field not in state:
                state[field] = base_val
                need_save = True
        
        now = time.time()
        if "turn_count" not in state:
            state["turn_count"] = 0
            need_save = True
        if "last_interaction" not in state:
            state["last_interaction"] = now
            need_save = True
        if "last_update" not in state:
            state["last_update"] = now
            need_save = True
        if "idle_penalty_applied" not in state:
            state["idle_penalty_applied"] = False
            need_save = True
        
        return need_save

    def get_user_state(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        key = f"{session_id}_{user_id}" if session_id else user_id
        if key not in self.user_data:
            lib_b, agg_b, slib_b, sagg_b = get_baseline(0.0)
            default = {
                "favour": 0.0,
                "attitude": self.DEFAULT_ATTITUDE,
                "relationship": self.DEFAULT_RELATIONSHIP,
                "libido": lib_b,
                "aggression": agg_b,
                "self_libido": slib_b,
                "self_aggression": sagg_b,
                "turn_count": 0,
                "last_interaction": time.time(),
                "last_update": time.time(),
                "idle_penalty_applied": False,
                "user_status":self.DEFAULT_STATUS,
            }
            return default
        return deepcopy(self.user_data[key])

    async def update_user_state(self, user_id: str, new_state: Dict[str, Any], session_id: Optional[str] = None):
        key = f"{session_id}_{user_id}" if session_id else user_id
        
        favour = max(-100.0, min(100.0, float(new_state.get("favour", 0.0))))
        new_state["favour"] = favour
        
        for dim in ["libido", "aggression", "self_libido", "self_aggression"]:
            if dim in new_state:
                new_state[dim] = clamp(new_state[dim], 0.0, 50.0)
        
        if "last_update" not in new_state:
            new_state["last_update"] = time.time()

        # 【修复死锁】：直接使用 to_thread 调用同步保存，不再调用 _save_data()
        async with self._lock:
            self.user_data[key] = new_state
            await asyncio.to_thread(self._save_sync, self.data_path / "user_data.json")

    async def batch_update(self, updates: Dict[str, Dict[str, Any]]):
        if not updates:
            return
        async with self._lock:
            for key, patch in updates.items():
                if key in self.user_data:
                    self.user_data[key].update(patch)
            await asyncio.to_thread(self._save_sync, self.data_path / "user_data.json")

    async def clear_all_data(self):
        """【新增】：安全的全局清理方法，避免外部调用死锁"""
        async with self._lock:
            self.user_data.clear()
            await asyncio.to_thread(self._save_sync, self.data_path / "user_data.json")

    def get_all_keys(self) -> list:
        return list(self.user_data.keys())

    def get_session_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取群聊/全局的环境情绪偏移量"""
        # 如果不开会话隔离，就当做是全宇宙共享一个大环境
        key = session_id if session_id else "global_universe"
        if key not in self.session_data:
            return {
                "libido_offset": 0.0,
                "aggression_offset": 0.0,
                "self_libido_offset": 0.0,
                "self_aggression_offset": 0.0,
                "last_update": time.time()
            }
        return deepcopy(self.session_data[key])

    async def update_session_state(self, session_id: Optional[str], new_state: dict):
        """保存环境情绪"""
        key = session_id if session_id else "global_universe"
        new_state["last_update"] = time.time()
        async with self._lock:
            self.session_data[key] = new_state
            # 存入独立的文件中
            path = self.data_path / "session_data.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)