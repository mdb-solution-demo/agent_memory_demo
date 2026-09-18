# -*- coding: utf-8 -*-
"""
游戏向记忆分类（OSS）：GAME_CUSTOM_CATEGORIES 定义维度与说明；
memory_metadata 为各维度 0~1 的浮点分数字段（0 最差，1 最好），数值统一保留小数点后 1 位。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from config import (
    MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_PREFIX,
    MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_SUFFIX,
)

logger = logging.getLogger(__name__)

# 与玩家当下状态相关的连续量表；与 memory_backends 事实抽取无关（事实仍走 mem0 抽取）
GAME_CUSTOM_CATEGORIES: List[dict[str, str]] = [
    {"task_priority": "玩家主观上对当前任务/目标的重视与急迫感（0=可忽略 1=非常看重且迫切）"},
    {"confidence": "玩家对完成当前任务或目标的信心（0=毫无信心 1=很有信心；如觉得没戏则 <0.5）"},
    {"emotion": "玩家当前整体情绪与心情倾向（0=很负面 1=很正面）"},
    {"npc_satisfaction": "玩家对当前 NPC 或本轮互动的满意程度（0=很不满意 1=很满意）"},
    {"engagement": "玩家对当前对话/游戏的投入与专注度（0=心不在焉 1=很投入）"},
]

CATEGORY_VALUE_KIND: Dict[str, str] = {}
for _item in GAME_CUSTOM_CATEGORIES:
    for _k in _item:
        CATEGORY_VALUE_KIND[_k] = "score"


def allowed_category_keys() -> List[str]:
    keys: List[str] = []
    for item in GAME_CUSTOM_CATEGORIES:
        keys.extend(item.keys())
    return keys


def default_category_key() -> str:
    keys = allowed_category_keys()
    return keys[0] if keys else "task_priority"


def is_allowed_category(key: str) -> bool:
    return key in set(allowed_category_keys())


def _clamp01(x: Any) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.5
    if f != f:  # NaN
        return 0.5
    return max(0.0, min(1.0, f))


def _score01(x: Any) -> float:
    """0~1 clamp 后保留一位小数。"""
    return round(_clamp01(x), 1)


def neutral_score_metadata() -> Dict[str, float]:
    return {k: _score01(0.5) for k in allowed_category_keys()}


def _category_descriptions_for_prompt() -> str:
    lines: List[str] = []
    for item in GAME_CUSTOM_CATEGORIES:
        for k, v in item.items():
            lines.append(f"  - {k} (0~1，保留一位小数): {v}")
    return "\n".join(lines)


def memory_metadata_value_is_present(val: Any) -> bool:
    """该槽位有内容：score 为 [0,1] 内数字；兼容旧版布尔 true、非空字符串、日期。"""
    if val is None or val is False:
        return False
    if isinstance(val, bool):
        return val is True
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val != val:
            return False
        return 0.0 <= float(val) <= 1.0
    if isinstance(val, str):
        return bool(val.strip())
    return True


def finalize_score_metadata(partial: Dict[str, Any]) -> Dict[str, float]:
    """合并 LLM/API 输出：仅允许键，clamp 到 [0,1] 并保留一位小数，缺失维度用 0.5。"""
    out = neutral_score_metadata()
    if not isinstance(partial, dict):
        return out
    for k, v in partial.items():
        if not is_allowed_category(k):
            continue
        if CATEGORY_VALUE_KIND.get(k) == "score":
            out[k] = _score01(v)
    return out


def normalize_memory_metadata_dict(raw: Dict[str, Any], seed_text: str) -> Dict[str, float]:
    """校验 API 覆盖传入的 memory_metadata，输出完整各维度 0~1 浮点（一位小数）。"""
    _ = seed_text  # 保留签名兼容调用方
    if not isinstance(raw, dict):
        return neutral_score_metadata()
    return finalize_score_metadata(dict(raw))


def _memory_metadata_extract_system_prompt() -> str:
    return (
        MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_PREFIX
        + _category_descriptions_for_prompt()
        + MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_SUFFIX
    )


def extract_memory_metadata_for_turn(
    user_message: str,
    npc_reply: str = "",
    *,
    api_key: str,
    model: str,
) -> Dict[str, float]:
    """
    用 LLM 抽取 memory_metadata；未配置 key 时返回全 0.5 中性量表（一位小数）。
    已配置 key 时，抽取失败则抛出异常，不再静默回退中性量表。
    """
    if not api_key:
        return neutral_score_metadata()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        if (npc_reply or "").strip():
            user_content = f"【玩家】\n{user_message}\n\n【NPC】\n{npc_reply}\n"
        else:
            user_content = f"【待归档内容】\n{user_message}\n"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _memory_metadata_extract_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            temperature=0.15,
            max_tokens=800,
            timeout=30.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            si = raw.find("{")
            ei = raw.rfind("}")
            if si >= 0 and ei > si:
                data = json.loads(raw[si : ei + 1])
            else:
                raise
        if not isinstance(data, dict):
            raise ValueError("not a dict")
        return finalize_score_metadata(data)
    except Exception as e:
        raise RuntimeError(f"memory_metadata LLM 抽取失败: {e}") from e


def item_metadata_matches_category(item: dict, category: str) -> bool:
    """
    该类别槽位「有内容」则命中；兼容旧数据顶层 memory_category、旧版 memory_metadata 布尔 true。
    """
    if not is_allowed_category(category):
        return False
    meta = item.get("metadata") or {}
    if meta.get("memory_category") == category:
        return True
    mm = meta.get("memory_metadata")
    if not isinstance(mm, dict):
        return False
    if category not in mm:
        return False
    return memory_metadata_value_is_present(mm.get(category))
