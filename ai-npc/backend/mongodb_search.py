# -*- coding: utf-8 -*-
"""
MongoDB / Atlas 检索统一入口。

两大域：
1. **extracted_memories（mem0）** — 仅 $vectorSearch（客户端嵌入 + 预过滤）
2. **game_wiki（百科）** — autoEmbed 向量 / Atlas Search 全文 / $rankFusion 混合
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from pymongo import MongoClient

from config import (
    GAME_WIKI_COLLECTION,
    GAME_WIKI_DB_NAME,
    GAME_WIKI_RANK_FUSION_SCORE_DETAILS,
    GAME_WIKI_RECALL_LIMIT,
    GAME_WIKI_SNIPPET_ANSWER_MAX,
    GAME_WIKI_TEXT_INDEX,
    GAME_WIKI_VECTOR_INDEX,
    GAME_WIKI_VECTOR_WEIGHT,
    MONGODB_URI,
    VOYAGE_API_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class VoyageRerankError(RuntimeError):
    """Voyage/MongoDB AI rerank 未配置 key、参数非法或远端失败。"""


class GameWikiSearchError(RuntimeError):
    """game_wiki 索引未就绪、URI 缺失或 Atlas 检索 / $rankFusion 失败。"""


# ---------------------------------------------------------------------------
# 常量 — mem0 文档字段
# ---------------------------------------------------------------------------

_NUM_CANDIDATES_CAP = 10_000

_PROMOTED_PAYLOAD_KEYS = ("user_id", "agent_id", "run_id", "actor_id", "role")
_CORE_PAYLOAD_KEYS = {"data", "hash", "created_at", "updated_at", "id", *_PROMOTED_PAYLOAD_KEYS}

# ---------------------------------------------------------------------------
# 常量 — Voyage rerank（仅环境变量，不在 config.py 中声明）
# ---------------------------------------------------------------------------

_VOYAGE_RERANK_MODEL = (os.getenv("VOYAGE_RERANK_MODEL") or "rerank-2.5").strip()
_VOYAGE_RERANK_DOC_MAX_CHARS = int(os.getenv("VOYAGE_RERANK_DOC_MAX_CHARS") or "4096")
_VOYAGE_RERANK_POOL_MULTIPLIER = max(1, int(os.getenv("VOYAGE_RERANK_POOL_MULTIPLIER") or "3"))
_VOYAGE_RERANK_POOL_MIN = max(1, int(os.getenv("VOYAGE_RERANK_POOL_MIN") or "10"))
_VOYAGE_RERANK_POOL_MAX = max(1, int(os.getenv("VOYAGE_RERANK_POOL_MAX") or "200"))

# ---------------------------------------------------------------------------
# 常量 — game_wiki $project
# ---------------------------------------------------------------------------

_GAME_WIKI_PROJECT: Dict[str, Any] = {
    "$project": {
        "_id": 1, "question": 1, "answer": 1, "category": 1,
        "page_name": 1, "component_name": 1, "conversations": 1, "score": 1,
    }
}

# ---------------------------------------------------------------------------
# 公共守卫工具
# ---------------------------------------------------------------------------


def _strip_query(query: Optional[str]) -> str:
    """返回 stripped query；空时返回 ""，调用方据此 early return。"""
    return (query or "").strip()


def _resolve_wiki_uri(mongo_uri: Optional[str]) -> str:
    """解析百科 URI：优先参数，回退 MONGODB_URI；空则返回 ""。"""
    return (mongo_uri or MONGODB_URI or "").strip()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _num_candidates(limit: int) -> int:
    """$vectorSearch numCandidates：20× limit，夹在 [limit, _NUM_CANDIDATES_CAP]。"""
    lim = max(1, int(limit))
    return min(max(lim * 20, lim), _NUM_CANDIDATES_CAP)


def _index_doc_queryable(doc: Optional[Dict[str, Any]]) -> bool:
    """判断单条索引元数据是否「可查询」。"""
    if not doc:
        return False
    st = (doc.get("status") or "").upper()
    if st and st != "READY":
        return False
    return doc.get("queryable") is not False


def _ensure_game_wiki_index_ready(collection: Any, index_name: str) -> None:
    """要求指定索引存在且 READY，否则抛 GameWikiSearchError。"""
    try:
        for doc in collection.list_search_indexes(name=index_name):
            if _index_doc_queryable(doc):
                return
    except Exception as e:
        raise GameWikiSearchError(
            f"list_search_indexes failed for {index_name}: {e}"
        ) from e
    raise GameWikiSearchError(
        f"game_wiki search index not READY or missing: {index_name}"
    )


def _build_vector_search_filter(
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    extra_filters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """构造 $vectorSearch 的 MQL filter（预过滤）。"""
    clauses: List[Dict[str, Any]] = []
    for field, val in (
        ("payload.user_id", user_id),
        ("payload.agent_id", agent_id),
        ("payload.run_id", run_id),
    ):
        if val is not None:
            clauses.append({field: val})
    if extra_filters:
        clauses.extend(
            {f"payload.{k}": v}
            for k, v in extra_filters.items()
            if v is not None and not isinstance(v, (dict, list))
        )
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _raw_doc_to_memory_item(doc: dict) -> dict:
    """MongoDB 原始文档 → 对外「记忆项」字典（与 mem0 语义对齐）。"""
    payload = doc.get("payload") or {}
    item: Dict[str, Any] = {
        "id": str(doc.get("_id")),
        "memory": payload.get("data", "") or "",
        "hash": payload.get("hash"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "score": doc.get("score"),
    }
    for key in _PROMOTED_PAYLOAD_KEYS:
        if key in payload:
            item[key] = payload[key]
    extra = {k: v for k, v in payload.items() if k not in _CORE_PAYLOAD_KEYS}
    if extra:
        item["metadata"] = extra
    return item


def _docs_to_mem0_results(docs: List[dict], threshold: Optional[float]) -> List[dict]:
    """向量检索结果 → 记忆项列表；可选按分数阈值过滤。"""
    out: List[dict] = []
    for doc in docs:
        item = _raw_doc_to_memory_item(doc)
        sc = item.get("score")
        if threshold is None or (isinstance(sc, (int, float)) and sc >= threshold):
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# 通用 Atlas 查询：vector / fulltext / rankFusion
# ---------------------------------------------------------------------------


def _build_atlas_search_body(
    *,
    text_search_index_name: str,
    user_query: str,
    text_search_paths: Optional[List[str]],
    text_search_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """构造 $search 阶段的索引体。"""
    if text_search_body is not None:
        body = dict(text_search_body)
        body.setdefault("index", text_search_index_name)
        return body
    if text_search_paths:
        q = (user_query or "").strip()
        if not q:
            raise ValueError("使用 text_search_paths 时必须提供非空 user_query")
        paths: Union[str, List[str]] = text_search_paths[0] if len(text_search_paths) == 1 else text_search_paths
        return {
            "index": text_search_index_name,
            "text": {"query": q, "path": paths},
        }
    raise ValueError("必须提供 text_search_body 或 text_search_paths")


def vector_search_with_mongodb(
    collection: Any,
    *,
    index_name: str,
    embedding_path: str = "embedding",
    limit: int = 10,
    query_vector: Optional[List[float]] = None,
    user_query: str = "",
    vector_filter: Optional[Dict[str, Any]] = None,
    result_project: Optional[Dict[str, Any]] = None,
    exclude_embedding: bool = True,
    score_meta: str = "vectorSearchScore",
) -> List[dict]:
    """通用 $vectorSearch：支持客户端 queryVector 或 autoEmbed query.text。"""
    lim = max(1, int(limit))
    if query_vector is None and not _strip_query(user_query):
        return []

    stage: Dict[str, Any] = {
        "index": index_name,
        "path": embedding_path,
        "numCandidates": _num_candidates(lim),
        "limit": lim,
    }
    if query_vector is not None:
        stage["queryVector"] = query_vector
    else:
        stage["query"] = {"text": (user_query or "").strip()}
    if vector_filter is not None:
        stage["filter"] = vector_filter

    pipeline: List[Dict[str, Any]] = [
        {"$vectorSearch": stage},
        {"$set": {"score": {"$meta": score_meta}}},
    ]
    if result_project is not None:
        pipeline.append({"$project": result_project})
    elif exclude_embedding:
        pipeline.append({"$project": {"embedding": 0}})

    return list(collection.aggregate(pipeline))


def fulltext_search_with_mongodb(
    collection: Any,
    *,
    text_search_index_name: str = "text_search_index",
    limit: int = 10,
    user_query: str = "",
    text_search_paths: Optional[List[str]] = None,
    text_search_body: Optional[Dict[str, Any]] = None,
    result_project: Optional[Dict[str, Any]] = None,
    exclude_embedding: bool = True,
    score_meta: str = "searchScore",
) -> List[dict]:
    """通用 Atlas $search（text_search_body 与 text_search_paths 二选一）。"""
    lim = max(1, int(limit))
    search_body = _build_atlas_search_body(
        text_search_index_name=text_search_index_name,
        user_query=user_query,
        text_search_paths=text_search_paths,
        text_search_body=text_search_body,
    )
    pipeline: List[Dict[str, Any]] = [
        {"$search": search_body},
        {"$limit": lim},
        {"$set": {"score": {"$meta": score_meta}}},
    ]
    if result_project is not None:
        pipeline.append({"$project": result_project})
    elif exclude_embedding:
        pipeline.append({"$project": {"embedding": 0}})

    return list(collection.aggregate(pipeline))


def hybrid_search_with_mongodb(
    user_query: str,
    collection: Any,
    *,
    vector_search_index_name: str = "vector_index",
    text_search_index_name: str = "text_search_index",
    vector_weight: float = 0.5,
    full_text_weight: float = 0.5,
    top_k: int = 10,
    per_pipeline_limit: Optional[int] = None,
    text_search_paths: Optional[List[str]] = None,
    text_search_body: Optional[Dict[str, Any]] = None,
    embedding_path: str = "embedding",
    query_vector: Optional[List[float]] = None,
    vector_filter: Optional[Dict[str, Any]] = None,
    score_details: bool = False,
    vector_pipeline_key: str = "vector",
    text_pipeline_key: str = "text",
    result_project: Optional[Dict[str, Any]] = None,
    exclude_embedding: bool = True,
    equal_weights: bool = False,
) -> List[dict]:
    """MongoDB 8.0+ $rankFusion 融合 $vectorSearch 与 $search。"""
    q = _strip_query(user_query)
    if not q:
        return []

    per_lim = max(1, int(per_pipeline_limit if per_pipeline_limit is not None else max(top_k * 2, 1)))
    top_k = max(1, int(top_k))

    search_body = _build_atlas_search_body(
        text_search_index_name=text_search_index_name,
        user_query=q,
        text_search_paths=text_search_paths,
        text_search_body=text_search_body,
    )

    vec_stage: Dict[str, Any] = {
        "index": vector_search_index_name,
        "path": embedding_path,
        "numCandidates": _num_candidates(per_lim),
        "limit": per_lim,
    }
    if query_vector is not None:
        vec_stage["queryVector"] = query_vector
    else:
        vec_stage["query"] = {"text": q}
    if vector_filter is not None:
        vec_stage["filter"] = vector_filter

    w_vec, w_txt = (1.0, 1.0) if equal_weights else (float(vector_weight), float(full_text_weight))

    pipeline: List[Dict[str, Any]] = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        vector_pipeline_key: [{"$vectorSearch": vec_stage}],
                        text_pipeline_key: [
                            {"$search": search_body},
                            {"$limit": per_lim},
                        ],
                    }
                },
                "combination": {
                    "weights": {vector_pipeline_key: w_vec, text_pipeline_key: w_txt}
                },
                "scoreDetails": bool(score_details),
            }
        },
        {"$limit": top_k},
        {"$set": {"score": {"$meta": "score"}}},
    ]
    if result_project is not None:
        pipeline.append({"$project": result_project})
    elif exclude_embedding:
        pipeline.append({"$project": {"embedding": 0}})

    return list(collection.aggregate(pipeline))


# ---------------------------------------------------------------------------
# extracted_memories（mem0）向量检索
# ---------------------------------------------------------------------------


def _require_scope_ids(
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
) -> None:
    """至少指定一个作用域 ID，避免全表扫描式检索。"""
    if not any([user_id, agent_id, run_id]):
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")


def _get_collection_and_indexes(memory: Any) -> tuple[Any, str]:
    """从 mem0 Memory 对象取底层 PyMongo Collection 与向量索引名。"""
    vs = memory.vector_store
    col = getattr(vs, "collection", None)
    if col is None:
        raise RuntimeError("MongoDB vector store collection is not initialized")
    return col, getattr(vs, "index_name", None) or f"{vs.collection_name}_vector_index"


def _embed_query(memory: Any, query: str) -> List[float]:
    """使用与入库相同的 embedding 模型将查询转为向量。"""
    embeddings = memory.embedding_model.embed(query, "search")
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return embeddings


def vector_search_memories(
    memory: Any,
    query: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    threshold: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """extracted_memories 纯向量检索：$vectorSearch + pre-filter。"""
    if not _strip_query(query):
        return {"results": []}
    _require_scope_ids(user_id, agent_id, run_id)
    col, index_name = _get_collection_and_indexes(memory)
    vfilter = _build_vector_search_filter(user_id, agent_id, run_id, dict(filters) if filters else None)
    qv = _embed_query(memory, query)
    vec_docs = vector_search_with_mongodb(
        col,
        index_name=index_name,
        embedding_path="embedding",
        limit=limit,
        query_vector=qv,
        vector_filter=vfilter,
        exclude_embedding=True,
        score_meta="vectorSearchScore",
    )
    return {"results": _docs_to_mem0_results(vec_docs[:limit], threshold)}


# =============================================================================
# game_wiki（百科）：独立库表；混合检索使用 MongoDB 8.0+ $rankFusion
# =============================================================================

_wiki_client_lock = threading.Lock()
_wiki_clients: Dict[str, MongoClient] = {}


def _get_game_wiki_client(mongo_uri: str) -> MongoClient:
    """懒加载并缓存与 URI 对应的 MongoClient。"""
    uri = mongo_uri.strip()
    pool = int(os.getenv("GAME_WIKI_MONGO_MAX_POOL_SIZE", "20"))
    with _wiki_client_lock:
        if uri not in _wiki_clients:
            _wiki_clients[uri] = MongoClient(uri, serverSelectionTimeoutMS=8000, maxPoolSize=max(1, pool))
        return _wiki_clients[uri]


def _game_wiki_collection(mongo_uri: str) -> Any:
    """返回 game_wiki 集合句柄。"""
    return _get_game_wiki_client(mongo_uri)[GAME_WIKI_DB_NAME][GAME_WIKI_COLLECTION]


def _game_wiki_lexical_compound(query: str) -> Dict[str, Any]:
    """百科 Atlas Search compound 子句（answer / question 两路 should）。"""
    return {
        "compound": {
            "should": [
                {"text": {"query": query, "path": "answer", "score": {"boost": {"value": 1.0}}}},
                {"text": {"query": query, "path": "question", "score": {"boost": {"value": 0.92}}}},
            ],
            "minimumShouldMatch": 1,
        }
    }


# --- 统一 wiki 检索入口 ---


# --- Voyage rerank 工具 ---


def voyage_rerank_pool_size(final_limit: int) -> int:
    """rerank 前先取的候选条数（记忆与 wiki 共用）。"""
    lim = max(1, int(final_limit))
    raw = max(lim * max(1, _VOYAGE_RERANK_POOL_MULTIPLIER), max(1, _VOYAGE_RERANK_POOL_MIN))
    cap = max(1, _VOYAGE_RERANK_POOL_MAX)
    return min(raw, cap, 1000)


def memory_item_text_for_rerank(item: dict) -> str:
    """mem0 检索结果项的纯文本，供全局 rerank。"""
    mx = max(256, _VOYAGE_RERANK_DOC_MAX_CHARS)
    m = (item.get("memory") or "").strip() if isinstance(item, dict) else ""
    return m[:mx] + "…" if len(m) > mx else m


def voyage_rerank_items_by_text(
    query: str,
    items: List[dict],
    *,
    top_k: int,
    text_from_item: Callable[[dict], str],
) -> List[dict]:
    """Voyage rerank：无 key / 空查询 / 远端失败时抛 VoyageRerankError。"""
    if not items:
        return []
    q = _strip_query(query)
    if not q:
        raise VoyageRerankError("rerank query is empty")
    key = (VOYAGE_API_KEY or "").strip()
    if not key:
        raise VoyageRerankError("VOYAGE_API_KEY is not set")
    tk = max(1, min(int(top_k), len(items)))
    try:
        import voyageai
        client = voyageai.Client(api_key=key)
        texts = [text_from_item(d) for d in items]
        rr = client.rerank(q, texts, model=_VOYAGE_RERANK_MODEL, top_k=tk, truncation=True)
        out: List[dict] = []
        for r in rr.results:
            if 0 <= r.index < len(items):
                d = dict(items[r.index])
                d["rerank_score"] = float(r.relevance_score)
                d["score"] = float(r.relevance_score)
                out.append(d)
        if not out:
            raise VoyageRerankError("Voyage rerank returned no scored results")
        return out
    except VoyageRerankError:
        raise
    except Exception as e:
        raise VoyageRerankError(f"Voyage rerank failed: {e}") from e


# --- wiki 文档 → 文本 ---


def _game_wiki_qa_from_doc(doc: dict) -> tuple[str, str]:
    """优先顶层 question/answer；否则回退到 conversations（旧文档）。"""
    if "question" in doc or "answer" in doc:
        return (
            ("" if (q := doc.get("question")) is None else str(q).strip()),
            ("" if (a := doc.get("answer")) is None else str(a).strip()),
        )
    conv = doc.get("conversations")
    if conv is None:
        conv = {}
    elif not isinstance(conv, dict):
        raise TypeError(f"conversations must be dict or None, got {type(conv).__name__}")
    return ((conv.get("question") or "").strip(), (conv.get("answer") or "").strip())


def _game_wiki_header_from_doc(doc: dict) -> str:
    """展示用标题：优先 category，否则 page_name | component_name。"""
    cat = doc.get("category")
    if cat is not None:
        return ", ".join(str(x) for x in cat[:8]) if isinstance(cat, list) else str(cat).strip()
    page = doc.get("page_name") or ""
    comp = doc.get("component_name") or ""
    return f"{page} | {comp}".strip(" |") if (page or comp) else ""


def _format_wiki_doc(doc: dict, max_answer_chars: int, *, prefix: str = "") -> str:
    """
    公共格式化：header + Q/A，answer 截断到 max_answer_chars。
    prefix 非空时作为行首标记（如 "[Wiki]"）。
    """
    header = _game_wiki_header_from_doc(doc)
    q, a = _game_wiki_qa_from_doc(doc)
    if len(a) > max_answer_chars:
        a = a[:max_answer_chars] + "…"
    line0 = f"{prefix} {header}".strip() if prefix else (header or "")
    if q:
        return f"{line0}\nQ: {q}\nA: {a}" if line0 else f"Q: {q}\nA: {a}"
    return f"{line0}\nA: {a}" if line0 else a


def _wiki_doc_text_for_rerank(doc: dict) -> str:
    """rerank 用纯文本（截断到 _VOYAGE_RERANK_DOC_MAX_CHARS）。"""
    mx = max(256, _VOYAGE_RERANK_DOC_MAX_CHARS)
    s = _format_wiki_doc(doc, mx)
    return s[:mx * 2] if len(s) > mx * 2 else s


# --- wiki 融合权重 ---


def _compute_wiki_weights(
    wiki_fusion: str,
    vector_weight: Optional[float],
    wiki_fts_weight: Optional[float],
) -> tuple[float, float, bool]:
    """返回 (w_vec, w_txt, equal_rrf)。"""
    fusion_u = (wiki_fusion or "RSF").strip().upper()
    if fusion_u == "RRF":
        return 0.5, 0.5, True
    if wiki_fts_weight is not None:
        fts = max(0.0, min(1.0, float(wiki_fts_weight)))
        return max(0.0, 1.0 - fts), fts, False
    w_vec = max(0.0, min(1.0, float(vector_weight if vector_weight is not None else GAME_WIKI_VECTOR_WEIGHT)))
    return w_vec, max(0.0, 1.0 - w_vec), False


def _wiki_ensure_indexes(col: Any, mode: str) -> None:
    """按 mode 检查所需的 Atlas 索引是否 READY。"""
    if mode in ("vector", "hybrid"):
        _ensure_game_wiki_index_ready(col, GAME_WIKI_VECTOR_INDEX)
    if mode in ("fulltext", "hybrid"):
        _ensure_game_wiki_index_ready(col, GAME_WIKI_TEXT_INDEX)


def search_game_wiki(
    query: str,
    *,
    mode: str = "hybrid",
    mongo_uri: Optional[str] = None,
    recall_limit: int = 10,
    vector_weight: Optional[float] = None,
    wiki_fusion: str = "RSF",
    wiki_fts_weight: Optional[float] = None,
    use_reranker: bool = False,
) -> List[dict]:
    """
    game_wiki 百科统一检索。

    mode:
      - "hybrid"   : $rankFusion 融合向量 + 全文（默认）
      - "vector"   : 仅 $vectorSearch（autoEmbed）
      - "fulltext"  : 仅 Atlas $search（BM25）

    hybrid 模式下 vector_weight / wiki_fusion / wiki_fts_weight 生效；
    use_reranker 对所有模式生效（需配置 VOYAGE_API_KEY）。
    """
    q = _strip_query(query)
    if not q:
        return []
    uri = _resolve_wiki_uri(mongo_uri)
    if not uri:
        return []

    lim = max(1, int(recall_limit))
    col = _game_wiki_collection(uri)
    _wiki_ensure_indexes(col, mode)

    try:
        if mode == "vector":
            results = vector_search_with_mongodb(
                col,
                index_name=GAME_WIKI_VECTOR_INDEX,
                embedding_path="answer",
                limit=lim,
                query_vector=None,
                user_query=q,
                result_project=_GAME_WIKI_PROJECT["$project"],
                exclude_embedding=False,
                score_meta="vectorSearchScore",
            )[:lim]

        elif mode == "fulltext":
            results = fulltext_search_with_mongodb(
                col,
                text_search_index_name=GAME_WIKI_TEXT_INDEX,
                limit=lim,
                text_search_body={"index": GAME_WIKI_TEXT_INDEX, **_game_wiki_lexical_compound(q)},
                result_project=_GAME_WIKI_PROJECT["$project"],
                exclude_embedding=False,
                score_meta="searchScore",
            )[:lim]

        else:  # hybrid
            w_vec, w_txt, equal_rrf = _compute_wiki_weights(wiki_fusion, vector_weight, wiki_fts_weight)
            pool = voyage_rerank_pool_size(lim) if use_reranker else lim
            per_pipe = max(pool * 2, 1)
            results = hybrid_search_with_mongodb(
                q, col,
                vector_search_index_name=GAME_WIKI_VECTOR_INDEX,
                text_search_index_name=GAME_WIKI_TEXT_INDEX,
                vector_weight=w_vec,
                full_text_weight=w_txt,
                top_k=pool,
                per_pipeline_limit=per_pipe,
                text_search_body={"index": GAME_WIKI_TEXT_INDEX, **_game_wiki_lexical_compound(q)},
                embedding_path="answer",
                query_vector=None,
                score_details=bool(GAME_WIKI_RANK_FUSION_SCORE_DETAILS),
                vector_pipeline_key="vector",
                text_pipeline_key="text",
                result_project=_GAME_WIKI_PROJECT["$project"],
                exclude_embedding=False,
                equal_weights=equal_rrf,
            )
    except VoyageRerankError:
        raise
    except Exception as e:
        raise GameWikiSearchError(f"game_wiki search failed (mode={mode}): {e}") from e

    if not use_reranker:
        return results
    return voyage_rerank_items_by_text(q, results, top_k=lim, text_from_item=_wiki_doc_text_for_rerank)


def search_game_wiki_formatted(
    query: str,
    *,
    mongo_uri: Optional[str] = None,
    recall_limit: Optional[int] = None,
    vector_weight: Optional[float] = None,
    wiki_fusion: str = "RSF",
    wiki_fts_weight: Optional[float] = None,
    use_reranker: bool = False,
    answer_max_len: Optional[int] = None,
) -> List[str]:
    """Chat 使用的百科入口：混合检索 → [Wiki] 片段字符串列表（去重）。"""
    q = _strip_query(query)
    if not q:
        return []
    lim = int(recall_limit if recall_limit is not None else max(8, GAME_WIKI_RECALL_LIMIT * 2))
    amax = int(answer_max_len if answer_max_len is not None else GAME_WIKI_SNIPPET_ANSWER_MAX)
    merged = search_game_wiki(
        q,
        mode="hybrid",
        mongo_uri=mongo_uri,
        recall_limit=lim,
        vector_weight=vector_weight,
        wiki_fusion=wiki_fusion,
        wiki_fts_weight=wiki_fts_weight,
        use_reranker=use_reranker,
    )

    out: List[str] = []
    seen: set[str] = set()
    for doc in merged:
        s = _format_wiki_doc(doc, amax, prefix="[Wiki]")
        key = " ".join(s.split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# 向后兼容别名
search_game_wiki_hybrid = search_game_wiki_formatted
