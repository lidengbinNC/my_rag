"""
Milvus 向量存储实现（企业级）

面试考点 & 设计决策：

1. 为什么选 Milvus 而非 FAISS？
   ┌──────────────┬──────────────────────────┬──────────────────────────┐
   │              │ FAISS                    │ Milvus                   │
   ├──────────────┼──────────────────────────┼──────────────────────────┤
   │ 部署方式      │ 进程内（内存）            │ 独立服务（可集群）         │
   │ 持久化        │ 手动序列化到磁盘          │ 原生持久化（WAL + S3）    │
   │ 元数据过滤    │ 手动实现（内存遍历）       │ 原生标量过滤（SQL-like）  │
   │ 扩展性        │ 单机，受内存限制          │ 分布式，支持亿级向量       │
   │ 高可用        │ 无                       │ 支持副本、Kafka 消息队列  │
   │ 并发安全      │ 需自行加锁               │ 内置事务和并发控制         │
   └──────────────┴──────────────────────────┴──────────────────────────┘

2. Collection Schema 设计（企业级关键）：
   - chunk_id (VARCHAR, PK)：业务主键，用于幂等 upsert
   - embedding (FLOAT_VECTOR)：向量字段
   - content (VARCHAR)：原始文本，避免二次查询数据库
   - knowledge_base_id (VARCHAR, indexed)：知识库隔离，支持多租户
   - document_id (VARCHAR, indexed)：文档级删除
   - source (VARCHAR)：来源文件名
   - created_at (INT64)：时间戳，支持时间范围过滤

3. 索引选型（HNSW vs IVF_FLAT）：
   - IVF_FLAT：倒排文件索引，nlist 个聚类中心，搜索时只扫描 nprobe 个桶
     适合：数据量 < 100 万，精度要求高
   - HNSW：分层可导航小世界图，搜索复杂度 O(log N)
     适合：数据量 > 100 万，低延迟要求，生产推荐
   - IVF_SQ8：量化压缩版 IVF，内存占用减少 4x，精度略降
     适合：内存受限场景

4. 幂等 Upsert：
   企业场景中文档可能被重复处理（重试、更新），使用 upsert 而非 insert
   避免重复向量导致检索结果重复

5. 连接管理：
   - 本地/私有化部署：host + port（gRPC）
   - Zilliz Cloud（托管 Milvus）：uri + token（HTTPS）
   - 连接池：pymilvus 内部管理，无需手动维护

6. 异步处理：
   pymilvus 2.x 的同步 SDK 用 asyncio.to_thread 包装，
   避免阻塞 FastAPI 的 event loop
"""

import asyncio
import time
from typing import Any

from my_rag.infrastructure.vector_store.base import BaseVectorStore, VectorSearchResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Milvus 字段名常量，避免魔法字符串
_FIELD_ID = "chunk_id"
_FIELD_EMBEDDING = "embedding"
_FIELD_SPARSE_EMBEDDING = "sparse_embedding"
_FIELD_CONTENT = "content"
_FIELD_KB_ID = "knowledge_base_id"
_FIELD_DOC_ID = "document_id"
_FIELD_SOURCE = "source"
_FIELD_CREATED_AT = "created_at"

# VARCHAR 最大长度（Milvus 限制）
_MAX_VARCHAR = 65535
_MAX_CONTENT_LEN = 4096
_MAX_ID_LEN = 256
_MAX_SOURCE_LEN = 512


class MilvusVectorStore(BaseVectorStore):
    """
    企业级 Milvus 向量存储

    生命周期：
    1. __init__：保存配置，不建立连接（延迟初始化）
    2. 首次操作时 _ensure_collection() 建立连接并创建/加载 Collection
    3. 应用退出时调用 close() 释放连接
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        enable_sparse_hybrid: bool = False,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        db_name: str = "default",
        uri: str = "",
        token: str = "",
        index_type: str = "HNSW",
        metric_type: str = "IP",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 256,
        hnsw_ef_search: int = 64,
    ):
        self._collection_name = collection_name
        self._dimension = dimension
        self._enable_sparse_hybrid = enable_sparse_hybrid
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._db_name = db_name
        self._uri = uri
        self._token = token
        self._index_type = index_type
        self._metric_type = metric_type
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        self._collection: Any = None  # pymilvus.Collection
        self._initialized = False
        self._has_sparse_field = False

    @property
    def supports_hybrid(self) -> bool:
        return self._enable_sparse_hybrid

    # ── 连接与 Collection 初始化 ─────────────────────────────────────────

    def _connect(self) -> None:
        """建立 Milvus 连接（同步，在线程中调用）"""
        from pymilvus import connections

        alias = "default"

        if self._uri:
            # Zilliz Cloud / 企业版 HTTPS 方式
            connections.connect(
                alias=alias,
                uri=self._uri,
                token=self._token,
                db_name=self._db_name,
            )
            logger.info("milvus_connected_via_uri", uri=self._uri[:40])
        else:
            # 本地 / 私有化部署 gRPC 方式
            kwargs: dict[str, Any] = {
                "alias": alias,
                "host": self._host,
                "port": self._port,
                "db_name": self._db_name,
            }
            if self._user:
                kwargs["user"] = self._user
                kwargs["password"] = self._password
            connections.connect(**kwargs)
            logger.info("milvus_connected", host=self._host, port=self._port)

    def _build_schema(self):
        """构建 Collection Schema（企业级字段设计）"""
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(
                name=_FIELD_ID,
                dtype=DataType.VARCHAR,
                max_length=_MAX_ID_LEN,
                is_primary=True,
                auto_id=False,          # 使用业务 ID，支持幂等 upsert
                description="Chunk UUID",
            ),
            FieldSchema(
                name=_FIELD_EMBEDDING,
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimension,
                description="Chunk embedding vector",
            ),
        ]

        if self._enable_sparse_hybrid:
            fields.append(
                FieldSchema(
                    name=_FIELD_SPARSE_EMBEDDING,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                    description="Chunk sparse embedding for hybrid retrieval",
                )
            )

        fields.extend([
            FieldSchema(
                name=_FIELD_CONTENT,
                dtype=DataType.VARCHAR,
                max_length=_MAX_CONTENT_LEN,
                description="Chunk text content",
            ),
            FieldSchema(
                name=_FIELD_KB_ID,
                dtype=DataType.VARCHAR,
                max_length=_MAX_ID_LEN,
                description="Knowledge base ID for multi-tenant isolation",
            ),
            FieldSchema(
                name=_FIELD_DOC_ID,
                dtype=DataType.VARCHAR,
                max_length=_MAX_ID_LEN,
                description="Document ID for document-level deletion",
            ),
            FieldSchema(
                name=_FIELD_SOURCE,
                dtype=DataType.VARCHAR,
                max_length=_MAX_SOURCE_LEN,
                description="Source file name",
            ),
            FieldSchema(
                name=_FIELD_CREATED_AT,
                dtype=DataType.INT64,
                description="Unix timestamp (seconds)",
            ),
        ])

        return CollectionSchema(
            fields=fields,
            description="RAG chunk vectors with metadata",
            enable_dynamic_field=True,  # 允许存储 schema 未定义的额外字段
        )

    def _build_dense_index_params(self) -> dict:
        """构建 Dense 向量索引参数"""
        if self._index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self._metric_type,
                "params": {
                    "M": self._hnsw_m,
                    "efConstruction": self._hnsw_ef_construction,
                },
            }
        elif self._index_type == "IVF_FLAT":
            return {
                "index_type": "IVF_FLAT",
                "metric_type": self._metric_type,
                "params": {"nlist": 1024},
            }
        elif self._index_type == "IVF_SQ8":
            return {
                "index_type": "IVF_SQ8",
                "metric_type": self._metric_type,
                "params": {"nlist": 1024},
            }
        else:
            return {
                "index_type": "FLAT",
                "metric_type": self._metric_type,
                "params": {},
            }

    @staticmethod
    def _build_sparse_index_params() -> dict:
        """构建 Sparse 向量索引参数。"""
        return {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {
                "drop_ratio_build": 0.0,
            },
        }

    def _ensure_collection(self) -> None:
        """确保 Collection 存在并已加载到内存（同步，在线程中调用）"""
        from pymilvus import Collection, utility

        self._connect()

        if not utility.has_collection(self._collection_name):
            schema = self._build_schema()
            collection = Collection(
                name=self._collection_name,
                schema=schema,
                consistency_level="Strong",   # 企业场景优先强一致性
            )
            # 为向量字段创建 ANN 索引
            collection.create_index(
                field_name=_FIELD_EMBEDDING,
                index_params=self._build_dense_index_params(),
            )
            if self._enable_sparse_hybrid:
                collection.create_index(
                    field_name=_FIELD_SPARSE_EMBEDDING,
                    index_params=self._build_sparse_index_params(),
                )
            # 为高频过滤字段创建标量索引，加速 WHERE 过滤
            collection.create_index(field_name=_FIELD_KB_ID, index_name="idx_kb_id")
            collection.create_index(field_name=_FIELD_DOC_ID, index_name="idx_doc_id")
            self._has_sparse_field = self._enable_sparse_hybrid

            logger.info(
                "milvus_collection_created",
                collection=self._collection_name,
                index_type=self._index_type,
            )
        else:
            collection = Collection(self._collection_name)
            field_names = {field.name for field in collection.schema.fields}
            self._has_sparse_field = _FIELD_SPARSE_EMBEDDING in field_names
            if self._enable_sparse_hybrid and not self._has_sparse_field:
                logger.warning(
                    "milvus_collection_missing_sparse_field",
                    collection=self._collection_name,
                    hint="recreate collection or disable RETRIEVAL_ENABLE_MILVUS_HYBRID",
                )

        # 加载 Collection 到内存（查询前必须 load）
        collection.load()
        self._collection = collection
        self._initialized = True
        logger.info(
            "milvus_collection_loaded",
            collection=self._collection_name,
            num_entities=collection.num_entities,
        )

    async def _get_collection(self):
        """获取 Collection（懒加载，线程安全）"""
        if not self._initialized:
            await asyncio.to_thread(self._ensure_collection)
        return self._collection

    # ── 核心 CRUD 操作 ────────────────────────────────────────────────────

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """批量 Upsert Dense 向量。"""
        collection = await self._get_collection()
        await self._upsert_rows(
            collection=collection,
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=None,
            texts=texts,
            metadatas=metadatas,
        )

    async def add_hybrid(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        sparse_embeddings: list[dict[int, float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """批量 Upsert Dense + Sparse 向量。"""
        collection = await self._get_collection()
        self._ensure_hybrid_capable()
        await self._upsert_rows(
            collection=collection,
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            texts=texts,
            metadatas=metadatas,
        )

    async def _upsert_rows(
        self,
        collection,
        ids: list[str],
        embeddings: list[list[float]],
        sparse_embeddings: list[dict[int, float]] | None,
        texts: list[str],
        metadatas: list[dict] | None,
    ) -> None:
        """
        批量 Upsert 向量（幂等操作）

        企业级关键点：
        - 使用 upsert 而非 insert，支持文档更新时重新索引
        - 分批写入，避免单次请求过大（Milvus 默认限制 16MB）
        - 截断超长文本，防止 VARCHAR 溢出
        """
        metas = metadatas or [{} for _ in ids]
        now = int(time.time())

        batch_size = 500  # Milvus 推荐单批不超过 1000 条
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i: i + batch_size]
            batch_embs = embeddings[i: i + batch_size]
            batch_texts = texts[i: i + batch_size]
            batch_metas = metas[i: i + batch_size]
            batch_sparse = sparse_embeddings[i: i + batch_size] if sparse_embeddings is not None else None

            data = [
                batch_ids,
                batch_embs,
            ]
            if self._has_sparse_field:
                data.append(batch_sparse or [{} for _ in batch_ids])
            data.extend([
                [t[:_MAX_CONTENT_LEN] for t in batch_texts],
                [m.get("knowledge_base_id", "")[:_MAX_ID_LEN] for m in batch_metas],
                [m.get("document_id", "")[:_MAX_ID_LEN] for m in batch_metas],
                [m.get("source", "")[:_MAX_SOURCE_LEN] for m in batch_metas],
                [now] * len(batch_ids),
            ])

            await asyncio.to_thread(collection.upsert, data)

        # 写入后 flush，确保数据持久化（生产环境可异步 flush）
        await asyncio.to_thread(collection.flush)

        logger.info(
            "milvus_vectors_upserted",
            count=len(ids),
            total=collection.num_entities,
            hybrid=self._has_sparse_field and sparse_embeddings is not None,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        """
        向量相似度检索（支持元数据过滤）

        企业级关键点：
        - expr 参数实现标量过滤（Milvus 原生支持，比 FAISS 手动过滤高效得多）
        - output_fields 指定返回字段，避免传输不必要数据
        - search_params ef 控制召回精度与速度的权衡
        """
        collection = await self._get_collection()

        # 构建标量过滤表达式（Milvus SQL-like 语法）
        expr = _build_filter_expr(filter_metadata)

        search_params = self._build_search_params()

        def _do_search():
            return collection.search(
                data=[query_embedding],
                anns_field=_FIELD_EMBEDDING,
                param=search_params,
                limit=top_k,
                expr=expr or None,
                output_fields=[
                    _FIELD_CONTENT, _FIELD_KB_ID, _FIELD_DOC_ID,
                    _FIELD_SOURCE, _FIELD_CREATED_AT,
                ],
                consistency_level="Bounded",  # 搜索允许有界一致性，降低延迟
            )

        results = await asyncio.to_thread(_do_search)
        return self._convert_hits(results)

    async def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse_embedding: dict[int, float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        ranker: str = "weighted",
        candidate_limit: int | None = None,
        rrf_k: int = 60,
    ) -> list[VectorSearchResult]:
        """Dense + Sparse 混合检索。"""
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        collection = await self._get_collection()
        self._ensure_hybrid_capable()

        expr = _build_filter_expr(filter_metadata)
        fetch_k = max(top_k, candidate_limit or top_k * 4)

        requests = [
            AnnSearchRequest(
                data=[query_embedding],
                anns_field=_FIELD_EMBEDDING,
                param=self._build_search_params(),
                limit=fetch_k,
                expr=expr or None,
            ),
            AnnSearchRequest(
                data=[query_sparse_embedding],
                anns_field=_FIELD_SPARSE_EMBEDDING,
                param=self._build_sparse_search_params(),
                limit=fetch_k,
                expr=expr or None,
            ),
        ]
        reranker = (
            RRFRanker(rrf_k)
            if ranker.lower() == "rrf"
            else WeightedRanker(dense_weight, sparse_weight)
        )

        def _do_search():
            return collection.hybrid_search(
                reqs=requests,
                rerank=reranker,
                limit=top_k,
                output_fields=[
                    _FIELD_CONTENT, _FIELD_KB_ID, _FIELD_DOC_ID,
                    _FIELD_SOURCE, _FIELD_CREATED_AT,
                ],
            )

        results = await asyncio.to_thread(_do_search)
        return self._convert_hits(results)

    async def delete(self, ids: list[str]) -> None:
        """按 chunk_id 批量删除"""
        collection = await self._get_collection()

        # Milvus 删除语法：主键 in [...]
        id_list = ", ".join(f'"{id_}"' for id_ in ids)
        expr = f'{_FIELD_ID} in [{id_list}]'

        await asyncio.to_thread(collection.delete, expr)
        await asyncio.to_thread(collection.flush)

        logger.info("milvus_vectors_deleted", count=len(ids))

    async def delete_by_metadata(self, key: str, value: str) -> None:
        """
        按元数据字段批量删除（企业高频操作：删除整个文档的所有 chunk）

        典型场景：用户删除文档 → delete_by_metadata("document_id", doc_id)
        """
        collection = await self._get_collection()

        # 只支持已建索引的字段（knowledge_base_id / document_id）
        allowed_fields = {_FIELD_KB_ID, _FIELD_DOC_ID}
        if key not in allowed_fields:
            logger.warning(
                "milvus_delete_unsupported_field",
                key=key,
                hint=f"Only {allowed_fields} support efficient deletion",
            )

        expr = f'{key} == "{value}"'
        await asyncio.to_thread(collection.delete, expr)
        await asyncio.to_thread(collection.flush)

        logger.info("milvus_vectors_deleted_by_metadata", key=key, value=value)

    def count(self) -> int:
        """返回 Collection 中的向量总数"""
        if self._collection is None:
            return 0
        return self._collection.num_entities

    async def close(self) -> None:
        """释放连接（应用退出时调用）"""
        from pymilvus import connections
        if self._collection is not None:
            await asyncio.to_thread(self._collection.release)
        await asyncio.to_thread(connections.disconnect, "default")
        self._initialized = False
        logger.info("milvus_disconnected")

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _build_search_params(self) -> dict:
        if self._index_type == "HNSW":
            return {"metric_type": self._metric_type, "params": {"ef": self._hnsw_ef_search}}
        elif self._index_type in ("IVF_FLAT", "IVF_SQ8"):
            return {"metric_type": self._metric_type, "params": {"nprobe": 16}}
        else:
            return {"metric_type": self._metric_type, "params": {}}

    @staticmethod
    def _build_sparse_search_params() -> dict:
        return {"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}

    def _ensure_hybrid_capable(self) -> None:
        if not self._has_sparse_field:
            raise RuntimeError(
                "Current Milvus collection does not have sparse field support. "
                "Recreate the collection or disable Milvus hybrid retrieval."
            )

    @staticmethod
    def _convert_hits(results) -> list[VectorSearchResult]:
        output: list[VectorSearchResult] = []
        if not results:
            return output

        for hit in results[0]:
            entity = hit.entity
            output.append(VectorSearchResult(
                chunk_id=hit.id,
                score=float(hit.score),
                content=entity.get(_FIELD_CONTENT, ""),
                metadata={
                    "knowledge_base_id": entity.get(_FIELD_KB_ID, ""),
                    "document_id": entity.get(_FIELD_DOC_ID, ""),
                    "source": entity.get(_FIELD_SOURCE, ""),
                    "created_at": entity.get(_FIELD_CREATED_AT, 0),
                },
            ))
        return output

    async def get_collection_stats(self) -> dict:
        """获取 Collection 统计信息（运维/监控用）"""
        collection = await self._get_collection()

        def _stats():
            return {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "schema": {f.name: str(f.dtype) for f in collection.schema.fields},
                "indexes": [idx.to_dict() for idx in collection.indexes],
            }

        return await asyncio.to_thread(_stats)


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _build_filter_expr(filter_metadata: dict | None) -> str:
    """
    将 Python dict 转换为 Milvus 过滤表达式

    面试考点：Milvus 支持 SQL-like 的标量过滤，可与向量检索联合使用
    支持：==, !=, >, <, >=, <=, in, not in, and, or

    Examples:
        {"knowledge_base_id": "kb-123"}
        → 'knowledge_base_id == "kb-123"'

        {"knowledge_base_id": "kb-123", "document_id": "doc-456"}
        → 'knowledge_base_id == "kb-123" and document_id == "doc-456"'
    """
    if not filter_metadata:
        return ""

    parts = []
    for key, value in filter_metadata.items():
        if isinstance(value, str):
            safe_val = value.replace('"', '\\"')
            parts.append(f'{key} == "{safe_val}"')
        elif isinstance(value, (int, float)):
            parts.append(f"{key} == {value}")
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                vals = ", ".join(f'"{v}"' for v in value)
            else:
                vals = ", ".join(str(v) for v in value)
            parts.append(f"{key} in [{vals}]")

    return " and ".join(parts)
