"""
文档管理接口

面试考点：
- BackgroundTasks 异步处理（对比 Java 的 @Async / CompletableFuture）
- 文件上传处理流程
- 分页查询分块内容
"""

import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.common import APIResponse, PageResult
from my_rag.api.schemas.document import DocumentResponse, DocumentListResponse, ChunkResponse
from my_rag.config.settings import settings
from my_rag.core.document_pipeline import process_document
from my_rag.infrastructure.database import KnowledgeBase, Document, Chunk, get_db
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}


def _to_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        content_type=doc.content_type,
        status=doc.status,
        chunk_count=doc.chunk_count,
        knowledge_base_id=doc.knowledge_base_id,
        created_at=doc.created_at,
    )


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=APIResponse[DocumentResponse],
)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传文档并自动触发后台解析 + 分块"""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix}，支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > settings.storage.max_file_size:
        raise HTTPException(status_code=400, detail="文件大小超过限制")

    upload_dir = settings.upload_path / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    file_path = upload_dir / f"{doc_id}{suffix}"
    file_path.write_bytes(content)

    doc = Document(
        id=doc_id,
        filename=filename,
        file_path=str(file_path),
        file_size=len(content),
        content_type=file.content_type,
        status="pending",
        knowledge_base_id=kb_id,
        created_at=datetime.now(),
    )
    db.add(doc)

    kb.document_count += 1
    kb.updated_at = datetime.now()

    await db.commit()

    logger.info("document_uploaded", doc_id=doc_id, filename=filename, size=len(content))

    # 同步处理文档（Phase 2 简单可靠，后续可改为 Celery 异步任务）
    await process_document(doc_id)

    # 重新加载以获取处理后的最新状态
    await db.refresh(doc)

    return APIResponse(data=_to_response(doc))


class BatchUploadResultItem(BaseModel):
    filename: str
    status: str          # "success" | "failed" | "skipped"
    doc_id: str = ""
    error: str = ""


class BatchUploadResponse(BaseModel):
    total: int
    success: int
    failed: int
    skipped: int
    results: list[BatchUploadResultItem]


@router.post(
    "/knowledge-bases/{kb_id}/documents/batch",
    response_model=APIResponse[BatchUploadResponse],
)
async def batch_upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    批量上传文档（支持同时上传多个文件）

    面试考点：
    - List[UploadFile] 接收多文件上传
    - 逐文件处理，单个失败不影响其他文件
    - 返回每个文件的处理结果，方便前端展示进度

    典型用途：将 HotpotQA context 导出的多个 docx 文件一次性上传到知识库
    """
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    results: list[BatchUploadResultItem] = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for file in files:
        filename = file.filename or "unknown"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if suffix not in ALLOWED_EXTENSIONS:
            results.append(BatchUploadResultItem(
                filename=filename,
                status="skipped",
                error=f"不支持的文件类型: {suffix}",
            ))
            skipped_count += 1
            continue

        try:
            content = await file.read()
            if len(content) > settings.storage.max_file_size:
                results.append(BatchUploadResultItem(
                    filename=filename,
                    status="skipped",
                    error="文件大小超过限制",
                ))
                skipped_count += 1
                continue

            upload_dir = settings.upload_path / kb_id
            upload_dir.mkdir(parents=True, exist_ok=True)

            doc_id = str(uuid.uuid4())
            file_path = upload_dir / f"{doc_id}{suffix}"
            file_path.write_bytes(content)

            doc = Document(
                id=doc_id,
                filename=filename,
                file_path=str(file_path),
                file_size=len(content),
                content_type=file.content_type,
                status="pending",
                knowledge_base_id=kb_id,
                created_at=datetime.now(),
            )
            db.add(doc)
            kb.document_count += 1
            kb.updated_at = datetime.now()
            await db.commit()

            logger.info("batch_document_uploaded", doc_id=doc_id, filename=filename, size=len(content))

            await process_document(doc_id)
            await db.refresh(doc)

            results.append(BatchUploadResultItem(
                filename=filename,
                status="success",
                doc_id=doc_id,
            ))
            success_count += 1

        except Exception as e:
            logger.error("batch_upload_failed", filename=filename, error=str(e))
            results.append(BatchUploadResultItem(
                filename=filename,
                status="failed",
                error=str(e),
            ))
            failed_count += 1

    logger.info(
        "batch_upload_completed",
        kb_id=kb_id,
        total=len(files),
        success=success_count,
        failed=failed_count,
        skipped=skipped_count,
    )

    return APIResponse(data=BatchUploadResponse(
        total=len(files),
        success=success_count,
        failed=failed_count,
        skipped=skipped_count,
        results=results,
    ))


@router.get(
    "/knowledge-bases/{kb_id}/documents",
    response_model=APIResponse[DocumentListResponse],
)
async def list_documents(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    result = await db.execute(
        select(Document)
        .where(Document.knowledge_base_id == kb_id)
        .order_by(Document.created_at.desc())
    )
    docs = [_to_response(d) for d in result.scalars().all()]
    return APIResponse(data=DocumentListResponse(items=docs, total=len(docs)))


@router.get(
    "/documents/{doc_id}/status",
    response_model=APIResponse[DocumentResponse],
)
async def get_document_status(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """查询文档处理状态（前端可轮询此接口）"""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return APIResponse(data=_to_response(doc))


@router.get(
    "/documents/{doc_id}/chunks",
    response_model=APIResponse[PageResult[ChunkResponse]],
)
async def list_document_chunks(
    doc_id: str,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """查看文档的分块结果"""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    offset = (page - 1) * page_size
    total_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)
    )
    total = total_result.scalar() or 0

    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(page_size)
    )
    chunks = [
        ChunkResponse(
            id=c.id,
            content=c.content,
            chunk_index=c.chunk_index,
            token_count=c.token_count,
            document_id=c.document_id,
        )
        for c in result.scalars().all()
    ]

    return APIResponse(
        data=PageResult(
            items=chunks,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )
    )


@router.delete(
    "/knowledge-bases/{kb_id}/documents/{doc_id}",
    response_model=APIResponse,
)
async def delete_document(
    kb_id: str,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    from pathlib import Path
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()

    kb = await db.get(KnowledgeBase, kb_id)
    if kb:
        kb.document_count = max(0, kb.document_count - 1)
        kb.chunk_count = max(0, kb.chunk_count - doc.chunk_count)
        kb.updated_at = datetime.now()

    await db.delete(doc)

    # 从 BM25 内存索引中移除该文档的所有 chunk（DB 级联删除由 ORM 处理）
    from my_rag.core.dependencies import get_sparse_retriever, get_vector_store
    get_sparse_retriever().remove_by_document_id(doc_id)

    # 删除整个文档的所有 chunk（更常用）
    milvus_store = get_vector_store()
    await milvus_store.delete_by_metadata("document_id", doc_id)
    return APIResponse(message="文档已删除")
