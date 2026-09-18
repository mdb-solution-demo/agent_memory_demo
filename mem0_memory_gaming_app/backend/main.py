# -*- coding: utf-8 -*-
"""
FastAPI 入口：POST /chat 接收玩家消息，召回记忆、用 DeepSeek 生成 NPC 回复、写入记忆。
并提供 GET /logs/stream 用于前端实时展示服务端日志。
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import (
    CHAT_HISTORY_MAX_MESSAGES,
    CHAT_HISTORY_MAX_MSG_CHARS,
    CLASSIFY_SYSTEM,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    GAME_THEME,
    GAME_WIKI_ENABLED,
    GAME_WIKI_RECALL_LIMIT,
    GAME_WIKI_RECALL_MAX_CHARS,
    MONGODB_URI,
    NPC_NAME,
    VOYAGE_API_KEY,
)
from custom_categories import (
    allowed_category_keys,
    extract_memory_metadata_for_turn,
    is_allowed_category,
    item_metadata_matches_category,
    normalize_memory_metadata_dict,
)
from db_init import run_all as db_init_run_all
from memory_backends import create_memory_for_backend
from mongodb_search import (
    GameWikiSearchError,
    VoyageRerankError,
    memory_item_text_for_rerank,
    search_game_wiki_formatted,
    vector_search_memories,
    voyage_rerank_items_by_text,
    voyage_rerank_pool_size,
)
from npc_personas import (
    FIXED_NPC_IDS,
    default_npc_id_for_theme,
    get_npc_system_prompt,
    list_npc_public_info,
    normalize_npc_id,
    npc_allowed_for_theme,
    validate_npc_id_for_chat,
)

logger = logging.getLogger(__name__)
_memory_cache: Dict[str, Any] = {}

# DeepSeek OpenAI 客户端单例（避免每次请求重建连接池和 SSL 上下文）
_deepseek_client: Optional[Any] = None


def _get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI
        _deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    return _deepseek_client

# 记忆写入策略：为避免重复/不一致，不在同一路径混用 infer=True/False
MEMORY_ADD_INFER = True

RECALL_LIMIT_LONG = 10
RECALL_LIMIT_SHORT = 10
RECALL_MAX_ITEMS = 15
RECALL_MAX_CHARS = 3000

# 日志缓冲：保留最近 N 行，供 SSE 推送给前端
LOG_BUFFER_MAX = 1000
_log_buffer: Deque[str] = deque(maxlen=LOG_BUFFER_MAX)
_log_lock = threading.Lock()


class LogBufferHandler(logging.Handler):
    """将日志写入 _log_buffer，便于 SSE 推送。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_lock:
                _log_buffer.append(msg)
        except Exception:
            pass


def _setup_log_stream():
    """把 LogBufferHandler 挂到 root logger，确保 INFO 级别日志能流入前端。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)  # 否则 root 默认 WARNING，会过滤掉 mem0/httpx 的 INFO
    h = LogBufferHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(h)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_log_stream()
    db_init_run_all()
    yield
    _memory_cache.clear()


app = FastAPI(title="NPC 对话记忆测试", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class ConversationHistoryMessage(BaseModel):
    """本轮 message 之前的 user/assistant 轮次（不含当前用户句）；与 Mem0 search/add 分离。"""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    npc_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None, description="当前会话 ID，用于短期记忆召回与写入")
    backend: str = Field(default="mongodb", description="仅支持 mongodb")
    short_term_expiration_days: Optional[int] = Field(
        default=7, ge=1, le=365,
        description="短期记忆保留天数，判定为 short_term 时写入 expiration_time = now + N days",
    )
    recall_memory_vector: bool = Field(default=True, description="是否召回长期/短期向量记忆")
    memory_vector_filter_mode: Literal["pre-filter", "post-filter"] = Field(
        default="pre-filter",
        description="pre-filter：自定义 $vectorSearch 预过滤；post-filter：mem0 原生 memory.search（默认 post-filter）",
    )
    recall_wiki_hybrid: bool = Field(default=True, description="是否检索 game_wiki 百科混合结果")
    wiki_fusion: Literal["RSF", "RRF"] = Field(
        default="RRF",
        description="RSF：加权 RRF；RRF：两路等权",
    )
    wiki_fts_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="全文路权重，取值 0～1（如 0.3）；RSF 时有效；RRF 时由后端忽略",
    )
    use_reranker: bool = Field(
        default=False,
        description="对向量记忆与 game_wiki 启用 Voyage rerank；需配置 VOYAGE_API_KEY，否则 400",
    )
    conversation_history: Optional[List[ConversationHistoryMessage]] = Field(
        default=None,
        description="当前 message 之前的对话（多轮连贯）；不含本轮用户输入。Mem0 retrieve/classify/add 仍仅用本轮 message。",
    )


class ChatResponse(BaseModel):
    reply: str
    recalled_memories: list = Field(default_factory=list)
    memory_type_this_turn: str = Field(default="long_term", description="本轮判定：long_term 或 short_term")
    backend: str


# ---------- 记忆管理 API 的请求/响应模型 ----------
class MemorySearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb", description="仅支持 mongodb")
    limit: int = Field(default=10, ge=1, le=100)
    memory_category: Optional[str] = Field(
        default=None,
        description="按 memory_metadata 该键存在且值非空过滤（服务端在结果侧匹配）",
    )
    use_reranker: bool = Field(
        default=False,
        description="对检索结果启用 Voyage rerank；需配置 VOYAGE_API_KEY，否则 400",
    )


class MemoryAddBody(BaseModel):
    """添加记忆（支持短期会话 session_id、短期过期、过程记忆）。"""
    content: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb")
    session_id: Optional[str] = Field(default=None, description="短期记忆（会话）：用 session_id 区分，会话结束可清空")
    expiration_days: Optional[int] = Field(default=None, ge=1, le=365, description="短期记忆（过期）：N 天后过期，不传则长期")
    memory_type: Optional[str] = Field(default=None, description="procedural_memory 表示过程记忆")
    memory_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="直接覆盖抽取结果；合法键见 GAME_CUSTOM_CATEGORIES，值为 0~1 浮点数（保留一位小数）；未填则走 LLM 抽取",
    )


class MemoryUpdateBody(BaseModel):
    new_content: str = Field(..., min_length=1)


class MemoryDeleteAllQuery(BaseModel):
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb")


def _get_memory(backend: str):
    if backend not in _memory_cache:
        _memory_cache[backend] = create_memory_for_backend(backend)
    return _memory_cache[backend]


def _mongodb_search_memories(
    memory: Any,
    query: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """extracted_memories：Atlas $vectorSearch 预过滤（vector_search_memories）。"""
    return vector_search_memories(
        memory,
        query,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        limit=limit,
        filters=filters,
    )


def _truncate_chat_message_text(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _normalize_conversation_history(
    history: Optional[List[ConversationHistoryMessage]],
) -> List[Dict[str, str]]:
    """清洗并截断：单条过长截断、条数过多丢弃头部（保留最近）。"""
    if not history:
        return []
    out: List[Dict[str, str]] = []
    for item in history:
        content = _truncate_chat_message_text(item.content, CHAT_HISTORY_MAX_MSG_CHARS)
        if not content:
            continue
        out.append({"role": item.role, "content": content})
    max_n = max(1, CHAT_HISTORY_MAX_MESSAGES)
    if len(out) > max_n:
        out = out[-max_n:]
    return out


def _build_npc_messages(
    system_context: str,
    history: Optional[List[ConversationHistoryMessage]],
    current_user_message: str,
) -> List[Dict[str, str]]:
    prior = _normalize_conversation_history(history)
    return [
        {"role": "system", "content": system_context},
        *prior,
        {"role": "user", "content": current_user_message},
    ]


def _npc_reply(
    user_message: str,
    recalled: list[str],
    npc_id: str,
    *,
    conversation_history: Optional[List[ConversationHistoryMessage]] = None,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_API_KEY，无法生成 NPC 回复",
        )
    client = _get_deepseek_client()
    system = get_npc_system_prompt(npc_id)

    user_memories = (
        "「召回的记忆」:\n" + "\n".join("- " + m for m in recalled)
    )
    system_context = system + "\n\n" + user_memories

    messages = _build_npc_messages(system_context, conversation_history, user_message)

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.3,
        timeout=30.0,
    )
    return (resp.choices[0].message.content or "").strip()


def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _parse_expiration_time(metadata: Any) -> Optional[datetime]:
    if not isinstance(metadata, dict):
        return None
    exp = metadata.get("expiration_time")
    if exp is None:
        return None
    if isinstance(exp, datetime):
        return exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
    if isinstance(exp, str):
        try:
            # supports ISO strings with/without timezone
            dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _filter_and_pack_recalled(results: list[dict], now: datetime, *, include_expired: bool) -> list[str]:
    seen: set[str] = set()
    packed: list[str] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        m = _normalize_memory_text(r.get("memory", "") or "")
        if not m or m in seen:
            continue
        exp = _parse_expiration_time(r.get("metadata"))
        if (not include_expired) and exp is not None and exp <= now:
            continue
        seen.add(m)
        packed.append(m)
    return packed


def _pack_recall_wiki_first(wiki_snippets: list[str], memory_lines: list[str]) -> list[str]:
    """
    wiki片段优先，再拼个人记忆；总条数/总字数受 RECALL_MAX_* 约束，
    wiki单独受 GAME_WIKI_RECALL_LIMIT / GAME_WIKI_RECALL_MAX_CHARS 约束。
    """
    out: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    wiki_n = 0

    for w in wiki_snippets:
        if wiki_n >= GAME_WIKI_RECALL_LIMIT:
            break
        t = _normalize_memory_text(w)
        if not t or t in seen:
            continue
        n = len(w) + 2
        if total_chars + n > GAME_WIKI_RECALL_MAX_CHARS:
            break
        if total_chars + n > RECALL_MAX_CHARS:
            break
        if len(out) >= RECALL_MAX_ITEMS:
            break
        seen.add(t)
        out.append(w)
        total_chars += n
        wiki_n += 1

    for m in memory_lines:
        if len(out) >= RECALL_MAX_ITEMS:
            break
        t = _normalize_memory_text(m)
        if not t or t in seen:
            continue
        n = len(m) + 2
        if total_chars + n > RECALL_MAX_CHARS:
            break
        seen.add(t)
        out.append(m)
        total_chars += n

    return out


def _memory_add_with_retry(memory: Any, messages: list[dict], *, infer: bool, kwargs: Dict[str, Any]) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            memory.add(messages, infer=infer, **kwargs)
            return
        except Exception as e:
            last_err = e
            # 简单退避，尽量吸收瞬时网络/LLM 抽取失败
            sleep_s = 0.2 * (2**attempt)
            time.sleep(sleep_s)
    raise last_err or RuntimeError("memory.add failed")


# ---------- 长短记忆分类：见 config.CLASSIFY_SYSTEM ----------


def _classify_memory_type(user_message: str) -> str:
    if not DEEPSEEK_API_KEY:
        return "long_term"
    try:
        client = _get_deepseek_client()
        content = f"【玩家】{user_message}\n"
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=200,
            timeout=15.0,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        return "short_term" if "short" in out else "long_term" if "long" in out else "zero_term"
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"记忆类型分类失败: {e}",
        ) from e


def _set_memory_metadata(meta: Dict[str, Any], mm: Dict[str, Any]) -> None:
    """写入 memory_metadata（各维度 0~1 浮点、保留一位小数，见 custom_categories）。"""
    meta["memory_metadata"] = mm


def _category_filters_for_backend(_backend: str, memory_category: Optional[str]) -> Tuple[Dict[str, Any], bool]:
    """
    (mem0 filters, need_python_category_filter)。
    富类型嵌套字段无法在 mem0 层做「有值」筛选，统一在结果侧用 item_metadata_matches_category。
    """
    if not memory_category or not str(memory_category).strip():
        return {}, False
    key = str(memory_category).strip()
    if not is_allowed_category(key):
        raise HTTPException(status_code=400, detail=f"无效的 memory_category: {key}")
    return {}, True


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    backend = (req.backend or "mongodb").strip().lower()
    if backend != "mongodb":
        raise HTTPException(status_code=400, detail="backend 仅支持 mongodb")
    user_id = (req.user_id or "player-1").strip()
    try:
        validate_npc_id_for_chat(req.npc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    npc_id = normalize_npc_id(req.npc_id)
    session_id = (req.session_id or "").strip() or None
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")
    if req.use_reranker and not (VOYAGE_API_KEY or "").strip():
        raise HTTPException(
            status_code=400,
            detail="启用 rerank 时需配置环境变量 VOYAGE_API_KEY",
        )

    # mem0初始化，获取记忆对象
    memory = _get_memory(backend)
    # 召回：长期记忆（无 run_id）+ 短期记忆（当前会话，run_id=session_id），合并去重
    now = datetime.now(timezone.utc)
    long_results: list[dict] = []
    short_results: list[dict] = []
    recall_mv = req.recall_memory_vector
    mem_filter_mode = req.memory_vector_filter_mode
    # post-filter 使用 mem0 原生 search；pre-filter 使用自定义 $vectorSearch 预过滤
    mongo_use_mem0_search = mem_filter_mode == "post-filter"

    # Step1：召回长期记忆（无 run_id）；开启 rerank 时先扩大候选池再截断到 RECALL_LIMIT_LONG
    mem_fetch_limit = voyage_rerank_pool_size(RECALL_LIMIT_LONG) if req.use_reranker else RECALL_LIMIT_LONG
    if recall_mv:
        try:
            if mongo_use_mem0_search:
                long_result = memory.search(
                    query=message,
                    user_id=user_id,
                    agent_id=npc_id,
                    limit=mem_fetch_limit,
                )
            else:
                long_result = _mongodb_search_memories(
                    memory,
                    message,
                    user_id=user_id,
                    agent_id=npc_id,
                    limit=mem_fetch_limit,
                )
            long_results = long_result.get("results", []) if isinstance(long_result, dict) else (long_result or [])
        except Exception as e:
            logger.error("长期记忆检索失败: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"记忆检索失败: {e}") from e
        if req.use_reranker and long_results:
            try:
                long_results = voyage_rerank_items_by_text(
                    message,
                    long_results,
                    top_k=RECALL_LIMIT_LONG,
                    text_from_item=memory_item_text_for_rerank,
                )
            except VoyageRerankError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e

    # Demo场景中暂时不需要
    # # Step2：召回短期记忆（当前会话：run_id=session_id）
    # if session_id:
    #     logger.info("当前会话ID: %s", session_id)
    # if recall_mv and session_id:
    #     try:
    #         if mongo_use_mem0_search:
    #             short_result = memory.search(
    #                 query=message,
    #                 user_id=user_id,
    #                 agent_id=npc_id,
    #                 run_id=session_id,
    #                 limit=RECALL_LIMIT_SHORT,
    #             )
    #         else:
    #             short_result = _mongodb_search_memories(
    #                 memory,
    #                 message,
    #                 user_id=user_id,
    #                 agent_id=npc_id,
    #                 run_id=session_id,
    #                 limit=RECALL_LIMIT_SHORT,
    #             )
    #         short_results = short_result.get("results", []) if isinstance(short_result, dict) else (short_result or [])
    #     except Exception as e:
    #         logger.warning("短期记忆检索失败: %s", e)

    # Step3：个人记忆去重列表（不预先截断条数，便于与百科拼接时再统一裁切）
    memory_lines: list[str] = []
    seen_mem: set[str] = set()
    for m in _filter_and_pack_recalled(long_results, now, include_expired=False) + _filter_and_pack_recalled(
        short_results, now, include_expired=False
    ):
        t = _normalize_memory_text(m)
        if not t or t in seen_mem:
            continue
        seen_mem.add(t)
        memory_lines.append(m)

    # 本轮写入分类（长/短/零）；需在百科检索前调用以便 zero_term 时跳过 wiki
    memory_type_this_turn = _classify_memory_type(message)

    # Step3b：MongoDB 下先混合检索 game_wiki 百科，再与记忆拼接。
    # zero_term 时不做百科混合检索（即使前端勾选 recall_wiki_hybrid）。
    wiki_snippets: list[str] = []
    if (
        req.recall_wiki_hybrid
        and GAME_WIKI_ENABLED
        and MONGODB_URI.strip()
    ):
        if memory_type_this_turn == "zero_term":
            logger.info("zero_term：没有关键信息的聊天，跳过 game_wiki 百科混合检索")
        else:
            try:
                wiki_snippets = search_game_wiki_formatted(
                    message,
                    mongo_uri=MONGODB_URI,
                    recall_limit=max(8, GAME_WIKI_RECALL_LIMIT * 2),
                    wiki_fusion=req.wiki_fusion,
                    wiki_fts_weight=req.wiki_fts_weight,
                    use_reranker=req.use_reranker,
                )
            except (VoyageRerankError, GameWikiSearchError) as e:
                raise HTTPException(status_code=502, detail=str(e)) from e

    recalled = _pack_recall_wiki_first(wiki_snippets, memory_lines)

    # Step4：生成回复（Mem0：retrieve/classify/add 仍仅用本轮 message；conversation_history 只供 LLM 多轮连贯）
    reply = _npc_reply(message, recalled, npc_id, conversation_history=req.conversation_history)

    # zero_term（纯问候/确认）无需抽取元数据或写入记忆，直接返回
    if memory_type_this_turn == "zero_term":
        return ChatResponse(
            reply=reply,
            recalled_memories=recalled,
            memory_type_this_turn=memory_type_this_turn,
            backend=backend,
        )

    messages = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]

    add_kwargs: Dict[str, Any] = {"user_id": user_id, "agent_id": npc_id}
    add_kwargs["metadata"] = add_kwargs.get("metadata") or {}

    # 根据对话内容(messages)，提取出记忆元数据，写入memory_metadata字段
    try:
        mm = extract_memory_metadata_for_turn(
            message, reply, api_key=DEEPSEEK_API_KEY or "", model=DEEPSEEK_MODEL
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    _set_memory_metadata(add_kwargs["metadata"], mm)

    # 如果是短期记忆，则需要加上一个过期时间expiration_time字段，方便mongodb TTL删除
    if memory_type_this_turn == "short_term":
        if session_id:
            add_kwargs["run_id"] = session_id
        # 为短期记忆写入 expiration_time（BSON Date 类型），值 = 当前时间 + 用户设定的保留天数
        retention_days = req.short_term_expiration_days or 7
        expiration_time = datetime.now(timezone.utc) + timedelta(days=retention_days)
        add_kwargs["metadata"]["expiration_time"] = expiration_time
        # logger.info(
        #     "短期记忆：保留 %d 天，expiration_time=%s (BSON Date)",
        #     retention_days, expiration_time.isoformat(),
        # )

    # Step5：写回记忆数据库，异步执行
    store_backend = backend
    store_messages = messages
    store_infer = MEMORY_ADD_INFER
    store_kwargs = dict(add_kwargs)

    def _store():
        mem = _get_memory(store_backend)
        _memory_add_with_retry(mem, store_messages, infer=store_infer, kwargs=store_kwargs)

    background_tasks.add_task(_store)

    return ChatResponse(
        reply=reply,
        recalled_memories=recalled,
        memory_type_this_turn=memory_type_this_turn,
        backend=backend,
    )

 

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
def serve_frontend():
    """浏览器打开 http://localhost:8000/ 时直接返回前端页面。"""
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html 不存在")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health():
    raw = (NPC_NAME or "").strip()
    theme_default = default_npc_id_for_theme()
    if raw in FIXED_NPC_IDS and npc_allowed_for_theme(raw):
        default_npc_id = raw
    else:
        default_npc_id = theme_default
    return {
        "status": "ok",
        "game_theme": GAME_THEME,
        "default_npc_id": default_npc_id,
    }


@app.get("/npcs")
def list_npcs() -> List[Dict[str, Any]]:
    """固定场景 NPC 列表（id、展示名、MBTI、特质），供前端与接口对齐。"""
    return list_npc_public_info()

# -----------------------------------------------------
# ---------- 界面上的记忆管理操作功能：搜索、按 ID 操作 ---------
# -----------------------------------------------------
def _norm_backend(backend: str) -> str:
    b = (backend or "mongodb").strip().lower()
    if b != "mongodb":
        raise HTTPException(status_code=400, detail="backend 仅支持 mongodb")
    return b


@app.get("/memory/by-session")
def memory_get_by_session(
    session_id: str = Query(..., min_length=1),
    user_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None, description="可选：限定 NPC/agent 范围，避免同 session_id 串用"),
    backend: str = Query("mongodb"),
    limit: int = Query(100, ge=1, le=500),
    include_expired: bool = Query(False, description="是否包含已过期的短期记忆（默认不包含）"),
    memory_category: Optional[str] = Query(
        None,
        description="按 memory_metadata 该键有非空内容过滤（服务端筛选）",
    ),
) -> Dict[str, Any]:
    """按 session_id（会话）获取短期记忆；可选 user_id 限定用户。mem0 内部用 run_id 存储。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs = {"run_id": session_id.strip(), "limit": limit}
        if user_id and str(user_id).strip():
            kwargs["user_id"] = str(user_id).strip()
        if agent_id and str(agent_id).strip():
            kwargs["agent_id"] = str(agent_id).strip()
        cat_f, need_cat_pf = _category_filters_for_backend(backend, memory_category)
        if cat_f:
            kwargs["filters"] = cat_f
        result = memory.get_all(**kwargs)
    except Exception as e:
        logger.warning("memory.get_all(by-session) 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    out = result if isinstance(result, dict) else {"results": result}
    results_list = out.get("results", []) if isinstance(out, dict) else []
    now = datetime.now(timezone.utc)
    filtered = []
    for item in results_list or []:
        if not isinstance(item, dict):
            continue
        exp = _parse_expiration_time(item.get("metadata"))
        if (not include_expired) and exp is not None and exp <= now:
            continue
        filtered.append(item)
    if need_cat_pf and (memory_category or "").strip():
        ck = str(memory_category).strip()
        filtered = [x for x in filtered if item_metadata_matches_category(x, ck)]
    out["results"] = filtered
    out["filtered_out_expired"] = (len(results_list or []) - len(filtered)) if not include_expired else 0
    return out


@app.get("/memory/by-user")
def memory_get_by_user(
    user_id: str = Query("player-1"),
    agent_id: Optional[str] = Query(None),
    backend: str = Query("mongodb"),
    limit: int = Query(100, ge=1, le=500),
    include_expired: bool = Query(False, description="是否包含已过期的短期记忆（默认不包含）"),
    memory_category: Optional[str] = Query(
        None,
        description="按 memory_metadata 该键有非空内容过滤（服务端筛选）",
    ),
) -> Dict[str, Any]:
    """按玩家 ID（及可选 agent_id）获取该范围下所有记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs = {"user_id": (user_id or "player-1").strip(), "limit": limit}
        if agent_id and str(agent_id).strip():
            kwargs["agent_id"] = str(agent_id).strip()
        cat_f, need_cat_pf = _category_filters_for_backend(backend, memory_category)
        if cat_f:
            kwargs["filters"] = cat_f
        result = memory.get_all(**kwargs)
    except Exception as e:
        logger.warning("memory.get_all 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    out = result if isinstance(result, dict) else {"results": result}
    results_list = out.get("results", []) if isinstance(out, dict) else []
    now = datetime.now(timezone.utc)
    filtered = []
    for item in results_list or []:
        if not isinstance(item, dict):
            continue
        exp = _parse_expiration_time(item.get("metadata"))
        if (not include_expired) and exp is not None and exp <= now:
            continue
        filtered.append(item)
    if need_cat_pf and (memory_category or "").strip():
        ck = str(memory_category).strip()
        filtered = [x for x in filtered if item_metadata_matches_category(x, ck)]
    out["results"] = filtered
    out["filtered_out_expired"] = (len(results_list or []) - len(filtered)) if not include_expired else 0
    return out


@app.post("/memory/search")
def memory_search(body: MemorySearchQuery) -> Dict[str, Any]:
    """搜索记忆：query + user_id + agent_id + limit；可选按 memory_metadata 类别命中过滤。"""
    if body.use_reranker and not (VOYAGE_API_KEY or "").strip():
        raise HTTPException(
            status_code=400,
            detail="启用 rerank 时需配置环境变量 VOYAGE_API_KEY",
        )
    backend = _norm_backend(body.backend)
    memory = _get_memory(backend)
    agent_id = (body.agent_id or "").strip() or None
    cat_f, need_cat_pf = _category_filters_for_backend(backend, body.memory_category)
    fetch_limit = voyage_rerank_pool_size(body.limit) if body.use_reranker else body.limit
    try:
        result = _mongodb_search_memories(
            memory,
            body.query,
            user_id=(body.user_id or "player-1").strip(),
            agent_id=agent_id,
            limit=fetch_limit,
            filters=cat_f if cat_f else None,
        )
    except Exception as e:
        logger.warning("memory.search 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    out = result if isinstance(result, dict) else {"results": result}
    results_list = out.get("results", []) if isinstance(out, dict) else []
    now = datetime.now(timezone.utc)
    # search 默认也过滤过期项（避免“短期记忆过期但仍命中搜索”）
    packed = _filter_and_pack_recalled(results_list, now, include_expired=False)
    packed_set = set(packed)
    # 保持 API 形状：返回原对象，但 results 里只保留未过期的项
    if isinstance(out, dict):
        filtered_items = []
        for item in results_list or []:
            if isinstance(item, dict):
                m = _normalize_memory_text(item.get("memory", "") or "")
                if m and m in packed_set:
                    filtered_items.append(item)
        if need_cat_pf and (body.memory_category or "").strip():
            ck = str(body.memory_category).strip()
            filtered_items = [x for x in filtered_items if item_metadata_matches_category(x, ck)]
        dropped_from_raw = len(results_list or []) - len(filtered_items)
        if body.use_reranker and filtered_items:
            try:
                filtered_items = voyage_rerank_items_by_text(
                    body.query,
                    filtered_items,
                    top_k=body.limit,
                    text_from_item=memory_item_text_for_rerank,
                )
            except VoyageRerankError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e
        elif len(filtered_items) > body.limit:
            filtered_items = filtered_items[: body.limit]
        out["results"] = filtered_items
        out["filtered_out_expired"] = dropped_from_raw
    return out


@app.get("/memory/{memory_id}")
def memory_get(
    memory_id: str,
    backend: str = Query("mongodb", description="仅支持 mongodb"),
) -> Dict[str, Any]:
    """按 ID 获取单条记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    try:
        out = memory.get(memory_id.strip())
    except Exception as e:
        logger.warning("memory.get 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    if out is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return out if isinstance(out, dict) else {"memory": out}


@app.patch("/memory/{memory_id}")
def memory_update(
    memory_id: str,
    body: MemoryUpdateBody,
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 ID 更新记忆内容（OSS 支持）。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    if not hasattr(memory, "update"):
        raise HTTPException(status_code=501, detail="当前后端不支持 update")
    try:
        memory.update(memory_id.strip(), body.new_content)
    except Exception as e:
        logger.warning("memory.update 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "memory_id": memory_id}


@app.delete("/memory/{memory_id}")
def memory_delete(
    memory_id: str,
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 ID 删除单条记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    try:
        memory.delete(memory_id.strip())
    except Exception as e:
        logger.warning("memory.delete 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "memory_id": memory_id}


@app.delete("/memory")
def memory_delete_all(
    user_id: str = Query("player-1"),
    agent_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None, description="短期会话：按 session_id 清空该会话记忆"),
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 user_id/agent_id 或 session_id 删除该范围下所有记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs: Dict[str, Any] = {"limit": 500}
        if session_id and str(session_id).strip():
            kwargs["run_id"] = str(session_id).strip()
            if user_id and str(user_id).strip():
                kwargs["user_id"] = str(user_id).strip()
            if agent_id and str(agent_id).strip():
                kwargs["agent_id"] = str(agent_id).strip()
        else:
            kwargs["user_id"] = (user_id or "player-1").strip()
            if agent_id and str(agent_id).strip():
                kwargs["agent_id"] = str(agent_id).strip()
        # mem0 delete_all 与 MongoDB vector_store.list() 返回格式不兼容（list 返回单层列表，
        # delete_all 却用 [0] 当列表，导致迭代到 tuple 报 'tuple' object has no attribute 'id'）
        # 改为：get_all 取回 id 列表，再逐条 delete。
        result = memory.get_all(**kwargs)
        results_list = result.get("results", result) if isinstance(result, dict) else result
        ids: List[str] = []
        for item in results_list or []:
            if isinstance(item, dict):
                ids.append(item.get("id"))
            elif hasattr(item, "id"):
                ids.append(getattr(item, "id"))
        for mid in ids:
            if mid:
                memory.delete(mid)
    except Exception as e:
        logger.warning("memory delete_all 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "deleted_count": len(ids)}


@app.post("/memory/add")
def memory_add(body: MemoryAddBody) -> Dict[str, Any]:
    """添加记忆；支持短期会话（session_id，内部传 run_id）、短期过期（expiration_days）、过程记忆（memory_type）。"""
    backend = _norm_backend(body.backend)
    memory = _get_memory(backend)
    user_id = (body.user_id or "player-1").strip()
    agent_id = (body.agent_id or "").strip() or None
    messages = [{"role": "user", "content": body.content}]
    kwargs: Dict[str, Any] = {"user_id": user_id, "agent_id": agent_id}
    if body.session_id and body.session_id.strip():
        kwargs["run_id"] = body.session_id.strip()
    kwargs["metadata"] = kwargs.get("metadata") or {}
    if body.memory_metadata is not None:
        _set_memory_metadata(
            kwargs["metadata"],
            normalize_memory_metadata_dict(body.memory_metadata, body.content),
        )
    else:
        try:
            mm = extract_memory_metadata_for_turn(
                body.content, "", api_key=DEEPSEEK_API_KEY or "", model=DEEPSEEK_MODEL
            )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        _set_memory_metadata(kwargs["metadata"], mm)
    if body.expiration_days is not None:
        # expiration_time 使用 datetime 对象，PyMongo 会自动存为 BSON Date 类型
        expiration_time = datetime.now(timezone.utc) + timedelta(days=body.expiration_days)
        kwargs["metadata"]["expiration_time"] = expiration_time
        # 同时保留易读的字符串字段供兼容
        kwargs["metadata"]["expiration_date"] = expiration_time.strftime("%Y-%m-%d")
    if body.memory_type:
        kwargs["memory_type"] = body.memory_type.strip()
    try:
        result = memory.add(messages, infer=MEMORY_ADD_INFER, **kwargs)
    except Exception as e:
        logger.warning("memory.add 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return result if isinstance(result, dict) else {"results": [result]}


async def _logs_sse_generator():
    """SSE 流：先发历史缓冲，再轮询新行并推送。"""
    with _log_lock:
        lines = list(_log_buffer)
    last_n = len(lines)
    for line in lines:
        yield f"data: {json.dumps({'line': line})}\n\n"
    while True:
        await asyncio.sleep(0.3)
        with _log_lock:
            new_lines = list(_log_buffer)
        if len(new_lines) > last_n:
            for line in new_lines[last_n:]:
                yield f"data: {json.dumps({'line': line})}\n\n"
            last_n = len(new_lines)


@app.get("/logs/stream")
async def logs_stream():
    """Server-Sent Events：实时推送服务端日志，供前端右侧面板展示。"""
    return StreamingResponse(
        _logs_sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
