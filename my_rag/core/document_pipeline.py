"""
文档处理流水线

面试考点：
- Pipeline 模式：将 Parse → Chunk → Embed → Index 编排为完整流水线
- 向量化后同时写入 FAISS (向量检索) 和 BM25 (关键词检索)
- 事务一致性：文档状态管理 + 异常回滚
"""

import json
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

            # 更新 BM25 索引（需要重建全量索引）
            await _rebuild_bm25_index(session, sparse_retriever)

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

            logger.info(
                "document_processed",
                document_id=document_id,
                chunk_count=len(text_chunks),
                status="completed",
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
    """从数据库加载全部 chunk 重建 BM25 索引"""
    from sqlalchemy import select
    result = await session.execute(select(Chunk))
    all_chunks = result.scalars().all()
    corpus = [
        {
            "id": c.id,
            "content": c.content,
            "source": json.loads(c.metadata_json).get("source", "") if c.metadata_json else "",
            "knowledge_base_id": c.knowledge_base_id,
        }
        for c in all_chunks
    ]
    sparse_retriever.build_index(corpus)
    logger.info("bm25_index_rebuilt", total_chunks=len(corpus))
