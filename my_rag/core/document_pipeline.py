"""
文档处理流水线

面试考点：
- Pipeline 模式：将 Parse → Chunk → Embed → Index 编排为完整流水线
- 向量化后同时写入 FAISS (向量检索) 和 BM25 (关键词检索)
- 事务一致性：文档状态管理 + 异常回滚
- 批量 ingest（batch_ingest_documents）：所有文档的 chunk 合并为一次 embed 调用，
  避免 BGE-M3 等本地模型每篇文档独立推理的固定启动开销
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

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


# ── 批量 ingest 进度跟踪 ──────────────────────────────────────────────
# 用内存字典存储各 task_id 的进度，轻量方案，无需引入 Redis/Celery
# key: task_id（由调用方生成，通常是 dataset_name）
# value: IngestProgress dataclass

@dataclass
class IngestProgress:
    task_id: str
    total: int
    done: int = 0
    failed: int = 0
    status: str = "running"   # running | completed | failed
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)


# 全局进度注册表（进程内单例）
_ingest_progress: dict[str, IngestProgress] = {}


def get_ingest_progress(task_id: str) -> Optional[IngestProgress]:
    return _ingest_progress.get(task_id)


async def batch_ingest_documents(
    doc_ids: list[str],
    task_id: str,
    concurrency: int = 4,
) -> IngestProgress:
    """
    批量处理多篇文档：Parse → Chunk → 合并 Embed → 写 Milvus/MySQL/BM25

    优化点（相比逐篇调用 process_document）：
    1. 所有文档的 chunk 合并为一次 embed_documents 调用，消除 BGE-M3 每次推理的固定启动开销
    2. 进度写入 _ingest_progress，前端可通过 /ingest-status/{task_id} 轮询
    3. 后台运行（由 BackgroundTasks 调用），HTTP 请求立即返回，不阻塞用户

    面试考点：
    - 批量 embed 的收益：BGE-M3 推理有固定 overhead（模型前向传播初始化），
      10 篇文章单独推理 = 10 × overhead，合并推理 = 1 × overhead
    - 进度跟踪用内存字典而非 DB，避免引入 Redis 依赖；重启后进度丢失可接受
    - concurrency 控制并发度，防止同时 parse/chunk 太多文档占满内存
    """
    progress = IngestProgress(task_id=task_id, total=len(doc_ids))
    _ingest_progress[task_id] = progress

    from my_rag.core.dependencies import get_embedding, get_vector_store, get_sparse_retriever

    embedding = get_embedding()
    vector_store = get_vector_store()
    sparse_retriever = get_sparse_retriever()

    # ── 阶段 1：并发 Parse + Chunk，收集所有 chunk ──────────────────────
    # 每篇文档独立 parse/chunk（CPU 密集，用 Semaphore 限制并发）
    sem = asyncio.Semaphore(concurrency)

    @dataclass
    class _DocChunks:
        doc_id: str
        chunk_ids: list[str]
        chunk_texts: list[str]
        chunk_metadatas: list[dict]
        text_chunks: list  # TextChunk 对象列表

    async def _parse_and_chunk(doc_id: str) -> Optional[_DocChunks]:
        async with sem:
            async with async_session_factory() as session:
                doc = await session.get(Document, doc_id)
                if not doc:
                    logger.error("batch_ingest_doc_not_found", doc_id=doc_id)
                    return None
                doc.status = "processing"
                await session.commit()

            try:
                parser = ParserFactory.get_parser(doc.filename)
                parsed = await asyncio.to_thread(parser.parse, doc.file_path)

                if not parsed.content.strip():
                    async with async_session_factory() as session:
                        d = await session.get(Document, doc_id)
                        if d:
                            d.status = "completed"
                            d.chunk_count = 0
                            await session.commit()
                    return None

                chunker = ChunkerFactory.create(
                    strategy=settings.chunk.strategy,
                    chunk_size=settings.chunk.size,
                    chunk_overlap=settings.chunk.overlap,
                )
                text_chunks = await asyncio.to_thread(
                    chunker.chunk,
                    parsed.content,
                    {
                        "source": doc.filename,
                        "document_id": doc.id,
                        "knowledge_base_id": doc.knowledge_base_id,
                        **parsed.metadata,
                    },
                )

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
                return _DocChunks(
                    doc_id=doc_id,
                    chunk_ids=chunk_ids,
                    chunk_texts=chunk_texts,
                    chunk_metadatas=chunk_metadatas,
                    text_chunks=text_chunks,
                )
            except Exception as e:
                logger.error("batch_ingest_parse_chunk_failed", doc_id=doc_id, error=str(e))
                async with async_session_factory() as session:
                    d = await session.get(Document, doc_id)
                    if d:
                        d.status = "failed"
                        await session.commit()
                return None

    parse_results = await asyncio.gather(*[_parse_and_chunk(did) for did in doc_ids])
    valid_results = [r for r in parse_results if r is not None]

    failed_parse = len(doc_ids) - len(valid_results)
    progress.failed += failed_parse

    if not valid_results:
        progress.status = "completed"
        progress.finished_at = time.time()
        return progress

    # ── 阶段 2：合并所有 chunk，一次批量 embed ──────────────────────────
    # 核心优化：消除每篇文档独立推理的 overhead
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []
    doc_chunk_ranges: list[tuple[str, int, int]] = []  # (doc_id, start_idx, end_idx)

    for r in valid_results:
        start = len(all_texts)
        all_texts.extend(r.chunk_texts)
        all_ids.extend(r.chunk_ids)
        all_metadatas.extend(r.chunk_metadatas)
        doc_chunk_ranges.append((r.doc_id, start, len(all_texts)))

    logger.info(
        "batch_ingest_embed_start",
        task_id=task_id,
        total_chunks=len(all_texts),
        total_docs=len(valid_results),
    )

    try:
        all_embeddings = await embedding.embed_documents(all_texts)
    except Exception as e:
        logger.error("batch_ingest_embed_failed", task_id=task_id, error=str(e))
        progress.failed += len(valid_results)
        progress.status = "failed"
        progress.error = str(e)
        progress.finished_at = time.time()
        return progress

    # ── 阶段 3：写 Milvus（一次批量 upsert）──────────────────────────────
    try:
        await vector_store.add(
            ids=all_ids,
            embeddings=all_embeddings,
            texts=all_texts,
            metadatas=all_metadatas,
        )
        VECTOR_STORE_SIZE.set(vector_store.count())
    except Exception as e:
        logger.error("batch_ingest_vector_store_failed", task_id=task_id, error=str(e))
        progress.failed += len(valid_results)
        progress.status = "failed"
        progress.error = str(e)
        progress.finished_at = time.time()
        return progress

    # ── 阶段 4：逐文档写 MySQL Chunk 表 + 更新 Document 状态 ─────────────
    bm25_chunks_all: list[dict] = []

    for r in valid_results:
        doc_id = r.doc_id
        start, end = next((s, e) for (d, s, e) in doc_chunk_ranges if d == doc_id)
        doc_embeddings_slice = all_embeddings[start:end]

        try:
            async with async_session_factory() as session:
                doc = await session.get(Document, doc_id)
                if not doc:
                    continue

                for i, tc in enumerate(r.text_chunks):
                    chunk = Chunk(
                        id=r.chunk_ids[i],
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
                doc.chunk_count = len(r.text_chunks)

                kb = await session.get(KnowledgeBase, doc.knowledge_base_id)
                if kb:
                    kb.chunk_count = (kb.chunk_count or 0) + len(r.text_chunks)
                    kb.updated_at = datetime.now()

                await session.commit()

            bm25_chunks_all.extend([
                {
                    "id": r.chunk_ids[i],
                    "content": tc.content,
                    "source": doc.filename,
                    "document_id": doc.id,
                    "knowledge_base_id": doc.knowledge_base_id,
                }
                for i, tc in enumerate(r.text_chunks)
            ])

            progress.done += 1
            DOC_CHUNK_COUNT.observe(len(r.text_chunks))

        except Exception as e:
            logger.error("batch_ingest_db_write_failed", doc_id=doc_id, error=str(e))
            progress.failed += 1

    # ── 阶段 5：BM25 批量追加（一次性，避免多次 add_chunks 重建）──────────
    if bm25_chunks_all:
        sparse_retriever.add_chunks(bm25_chunks_all)

    EMBEDDING_BATCH_SIZE.observe(len(all_texts))
    progress.status = "completed"
    progress.finished_at = time.time()

    logger.info(
        "batch_ingest_done",
        task_id=task_id,
        total=len(doc_ids),
        done=progress.done,
        failed=progress.failed,
        elapsed_s=progress.elapsed_s,
    )
    return progress


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
