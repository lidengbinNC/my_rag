"""文档管理接口"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.common import APIResponse
from my_rag.api.schemas.document import DocumentResponse, DocumentListResponse
from my_rag.config.settings import settings
from my_rag.infrastructure.database import KnowledgeBase, Document, get_db
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt", ".html"}


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
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix}，支持: {', '.join(ALLOWED_EXTENSIONS)}",
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
    await db.flush()

    logger.info("document_uploaded", doc_id=doc_id, filename=filename, size=len(content))

    return APIResponse(data=_to_response(doc))


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
        kb.updated_at = datetime.now()

    await db.delete(doc)
    return APIResponse(message="文档已删除")
