"""知识库管理接口"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.common import APIResponse, PageResult
from my_rag.api.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
)
from my_rag.infrastructure.database import KnowledgeBase, get_db

router = APIRouter(prefix="/knowledge-bases")


def _to_response(kb: KnowledgeBase) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        document_count=kb.document_count,
        chunk_count=kb.chunk_count,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.post("", response_model=APIResponse[KnowledgeBaseResponse])
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
):
    kb = KnowledgeBase(
        id=str(uuid.uuid4()),
        name=body.name,
        description=body.description,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    db.add(kb)
    await db.flush()
    return APIResponse(data=_to_response(kb))


@router.get("", response_model=APIResponse[PageResult[KnowledgeBaseResponse]])
async def list_knowledge_bases(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    total_result = await db.execute(select(func.count(KnowledgeBase.id)))
    total = total_result.scalar() or 0

    result = await db.execute(
        select(KnowledgeBase)
        .order_by(KnowledgeBase.updated_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = [_to_response(kb) for kb in result.scalars().all()]

    return APIResponse(
        data=PageResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )
    )


@router.get("/{kb_id}", response_model=APIResponse[KnowledgeBaseResponse])
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return APIResponse(data=_to_response(kb))


@router.put("/{kb_id}", response_model=APIResponse[KnowledgeBaseResponse])
async def update_knowledge_base(
    kb_id: str,
    body: KnowledgeBaseUpdate,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    kb.updated_at = datetime.now()
    await db.flush()

    return APIResponse(data=_to_response(kb))


@router.delete("/{kb_id}", response_model=APIResponse)
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    await db.delete(kb)
    return APIResponse(message="已删除")
