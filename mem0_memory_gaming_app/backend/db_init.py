# -*- coding: utf-8 -*-
"""
数据库初始化模块（db_init.py）
==============================

应用启动时自动检查并创建「存记忆」所需的库、表、索引：
- MongoDB：确保有向量搜索索引，否则无法按相似度查记忆

逻辑：存在就跳过，不存在才创建，避免重复创建报错。
"""

import logging

# 从 config 里拿库名/集合名与「向量维度」，与 memory_backends 共用同一来源
from config import (
    EMBEDDING_DIMS,
    MONGODB_COLLECTION_NAME,
    MONGODB_DB_NAME,
    MONGODB_URI,
)

# 用 Python 自带的 logging 打日志，方便排查问题；__name__ 会显示是 db_init 模块在打
logger = logging.getLogger(__name__)

# 与 mem0 约定一致：向量索引名 = {collection}_vector_index；全文索引供混合检索
MONGODB_VECTOR_INDEX_NAME = f"{MONGODB_COLLECTION_NAME}_vector_index"
MONGODB_TEXT_INDEX_NAME = f"{MONGODB_COLLECTION_NAME}_text_index"


def ensure_mongodb_indexes() -> None:
    """
    确保 MongoDB 里存在「向量搜索索引」。
    - 没配置 MONGODB_URI 则直接返回，不报错
    - 已有索引则只打日志「已存在」
    - 没有索引则创建（Atlas 上创建后可能要等约 1 分钟才真正可用）
    """
    # 未配置或配成空字符串，就不做任何事
    if not MONGODB_URI or not MONGODB_URI.strip():
        logger.info("MONGODB_URI 未配置，跳过 MongoDB 资源初始化")
        return
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel

    # 连上 MongoDB（Atlas 会按 URI 自动选集群）
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    # Atlas Search / Vector Search 要求集合已存在；空库时需先建集合
    if MONGODB_COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(MONGODB_COLLECTION_NAME)
        logger.info("MongoDB 集合已创建: %s.%s", MONGODB_DB_NAME, MONGODB_COLLECTION_NAME)
    col = db[MONGODB_COLLECTION_NAME]
    # 向量索引名与 mem0 MongoDBVectorStore 一致，否则 mem0 检索会报「索引不存在」
    index_name = MONGODB_VECTOR_INDEX_NAME

    existing = list(col.list_search_indexes(name=index_name))
    if existing:
        logger.info("MongoDB 向量索引已存在: %s", index_name)
    else:
        # mem0 文档形态：{ embedding, payload:{ user_id, agent_id, run_id, data, ... } }
        col.create_search_index(
            SearchIndexModel(
                name=index_name,
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIMS,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "payload.user_id"},
                        {"type": "filter", "path": "payload.agent_id"},
                        {"type": "filter", "path": "payload.run_id"},
                    ]
                },
                type="vectorSearch",
            )
        )
        logger.info(
            "MongoDB 向量索引已创建: %s（Atlas 上可能需要约 1 分钟就绪）", index_name
        )

    client.close()


def run_all() -> None:
    """
    执行「全部」初始化
    应用启动时在 main.py 的 lifespan 里调用一次即可。
    """
    ensure_mongodb_indexes()
