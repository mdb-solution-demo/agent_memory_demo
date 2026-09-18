# -*- coding: utf-8 -*-
"""
记忆后端：构造基于 MongoDB 的 mem0 Memory。
mem0 内部用 LLM 做事实抽取，这里统一用 DeepSeek；NPC 回复与长短记忆分类也由 main.py 使用 DeepSeek。
"""

import os

from langchain_voyageai import VoyageAIEmbeddings
from config import (
    CUSTOM_FACT_EXTRACTION_PROMPT,
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_COLLECTION_NAME,
    VOYAGE_API_KEY,
    EMBEDDING_DIMS,
    DEEPSEEK_API_KEY,
)


def _voyage_embedder():
    return VoyageAIEmbeddings(model="voyage-4-lite", voyage_api_key=VOYAGE_API_KEY)


def get_mongodb_config():
    return {
        "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
        "vector_store": {
            "provider": "mongodb",
            "config": {
                "db_name": MONGODB_DB_NAME,
                "collection_name": MONGODB_COLLECTION_NAME,
                "mongo_uri": MONGODB_URI,
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
        "embedder": {
            "provider": "langchain",
            "config": {"model": _voyage_embedder(), "embedding_dims": EMBEDDING_DIMS},
        },
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 1.0,
                "api_key": DEEPSEEK_API_KEY,
            },
        },
    }


def create_memory_for_backend(backend: str = "mongodb"):
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "true")
    from mem0 import Memory

    if (backend or "mongodb").strip().lower() != "mongodb":
        raise ValueError(f"Unsupported backend: {backend} (only mongodb)")
    return Memory.from_config(get_mongodb_config())
