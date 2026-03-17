"""
文档处理流水线

面试考点：
- Pipeline 模式：将 Parse → Chunk → Embed → Index 编排为完整流水线
- 向量化后同时写入 FAISS (向量检索) 和 BM25 (关键词检索)
- 事务一致性：文档状态管理 + 异常回滚
"""

import json
import time
import uuid
from datetime import datetime

from my_rag.config.settings import settings
from my_rag.domain.chunking.factory import ChunkerFactory
from my_rag.domain.parser.factory import ParserFactory
from my_rag.infrastructure.database import (
    Chunk,
    Document,
    KnowledgeBase,
    async_session_factory,
)
from my_rag.utils.logger import get_logger
from my_rag.utils.metrics import DOC_PROCESS_DURATION, DOC_CHUNK_COUNT, EMBEDDING_BATCH_SIZE, VECTOR_STORE_SIZE

logger = get_logger(__name__)


async def process_document(document_id: str) -> None:
    """
    后台文档处理入口

    使用独立的数据库会话，与 HTTP 请求的 session 解耦。
    这是因为 BackgroundTask 在 Response 发送后执行，
    原始请求的 DB session 此时已关闭。
    """
    async with async_session_factory() as session:
        doc = await session.get(Document, document_id)
        if not doc:
            logger.error("document_not_found", document_id=document_id)
            return

        doc.status = "processing"
        await session.commit()

        _start = time.perf_counter()
        try:
            parser = ParserFactory.get_parser(doc.filename)
            parsed = parser.parse(doc.file_path)

            logger.info(
                "document_parsed",
                document_id=document_id,
                content_length=len(parsed.content),
                metadata=parsed.metadata,
            )

            if not parsed.content.strip():
                doc.status = "completed"
                doc.chunk_count = 0
                await session.commit()
                logger.warning("document_empty", document_id=document_id)
                return

            chunker = ChunkerFactory.create(
                strategy=settings.chunk.strategy,
                chunk_size=settings.chunk.size,
                chunk_overlap=settings.chunk.overlap,
            )

            text_chunks = chunker.chunk(
                parsed.content,
                metadata={
                    "source": doc.filename,
                    "document_id": doc.id,
                    "knowledge_base_id": doc.knowledge_base_id,
                    **parsed.metadata,
                },
            )

            logger.info(
                "document_chunked",
                document_id=document_id,
                chunk_count=len(text_chunks),
                strategy=chunker.strategy_name,
            )

            # === Embedding + 向量索引 ===
            from my_rag.core.dependencies import get_embedding, get_vector_store, get_sparse_retriever

            embedding = get_embedding()
            vector_store = get_vector_store()
            sparse_retriever = get_sparse_retriever()

            chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
            chunk_texts = [tc.content for tc in text_chunks]
            chunk_metadatas = [
                {
                    "source": doc.filename,
                    "document_id": doc.id,
                    "knowledge_base_id": doc.knowledge_base_id,
                }
                for _ in text_chunks
            ]

            embeddings = await embedding.embed_documents(chunk_texts)

            logger.info(
                "document_embedded",
                document_id=document_id,
                chunk_count=len(embeddings),
            )

            await vector_store.add(
                ids=chunk_ids,
                embeddings=embeddings,
                texts=chunk_texts,
                metadatas=chunk_metadatas,
            )
            VECTOR_STORE_SIZE.set(vector_store.count())

            for i, tc in enumerate(text_chunks):
                chunk = Chunk(
                    id=chunk_ids[i],
                    content=tc.content,
                    chunk_index=tc.chunk_index,
                    token_count=tc.token_count,
                    metadata_json=json.dumps(tc.metadata, ensure_ascii=False),
                    document_id=doc.id,
                    knowledge_base_id=doc.knowledge_base_id,
                    created_at=datetime.now(),
                )
                session.add(chunk)

            doc.status = "completed"
            doc.chunk_count = len(text_chunks)

            kb = await session.get(KnowledgeBase, doc.knowledge_base_id)
            if kb:
                kb.chunk_count = (kb.chunk_count or 0) + len(text_chunks)
                kb.updated_at = datetime.now()

            await session.commit()

            # BM25 增量追加：commit 之后执行，无需查 DB，直接用本次已处理的 chunk
            # 面试考点：增量追加 vs 全量重建
            #   - 旧做法：每次 commit 前查全表 SELECT * FROM chunks，对所有 chunk 重新分词
            #   - 新做法：只对本次新增的 chunk 分词并追加，O(新增) 而非 O(全量)
            #   - 启动时仍用 _rebuild_bm25_index 从 DB 全量恢复（冷启动场景）
            new_bm25_chunks = [
                {
                    "id": chunk_ids[i],
                    "content": tc.content,
                    "source": doc.filename,
                    "document_id": doc.id,
                    "knowledge_base_id": doc.knowledge_base_id,
                }
                for i, tc in enumerate(text_chunks)
            ]
            sparse_retriever.add_chunks(new_bm25_chunks)

            DOC_PROCESS_DURATION.observe(time.perf_counter() - _start)
            DOC_CHUNK_COUNT.observe(len(text_chunks))
            EMBEDDING_BATCH_SIZE.observe(len(chunk_texts))

            logger.info(
                "document_processed",
                document_id=document_id,
                chunk_count=len(text_chunks),
                status="completed",
                duration_s=round(time.perf_counter() - _start, 2),
            )

        except Exception as e:
            logger.error(
                "document_process_failed",
                document_id=document_id,
                error=str(e),
            )
            doc.status = "failed"
            await session.commit()


async def _rebuild_bm25_index(session, sparse_retriever) -> None:
    """
    冷启动时从 DB 全量重建 BM25 索引。

    调用时机：
    - 应用启动（main.py lifespan）：进程重启后内存索引丢失，需从 DB 恢复
    - 不在每次文档处理后调用（改为增量追加 sparse_retriever.add_chunks）

    面试考点（正排索引 vs 倒排索引）：
    - 正排索引：doc_id → [term1, term2, ...]，适合
    "给定文档，查它包含哪些词"
    - 倒排索引：term → [doc_id1, doc_id2, ...]，适合"给定关键词，查哪些文档包含它"
    - BM25 基于倒排索引，rank_bm25 在内存中维护词频矩阵，本质是稠密倒排索引
    """
    from sqlalchemy import select

    result = await session.execute(select(Chunk))
    all_chunks = result.scalars().all()

    corpus = [
        {
            "id": c.id,
            "content": c.content,
            "source": json.loads(c.metadata_json).get("source", "") if c.metadata_json else "",
            "document_id": c.document_id,
            "knowledge_base_id": c.knowledge_base_id,
        }
        for c in all_chunks
    ]

    sparse_retriever.build_index(corpus)
    logger.info("bm25_index_rebuilt", total_chunks=len(corpus))
