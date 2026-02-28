"""
文档处理流水线

面试考点：
- Pipeline 模式：将 Parse → Chunk 编排为可组合的流水线
- 后台任务处理：FastAPI BackgroundTasks（轻量）vs Celery（重量）
- 事务一致性：文档状态管理 + 异常回滚
- 独立数据库会话：后台任务不共享请求的 session
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

            for tc in text_chunks:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
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
