"""知识库管理接口"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.common import APIResponse, PageResult
from my_rag.api.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from my_rag.domain.knowledge import loads_str_list, profile_matches
from my_rag.infrastructure.database import KnowledgeBase, KnowledgeBaseProfile, get_db

router = APIRouter(prefix="/knowledge-bases")


async def _get_profile(db: AsyncSession, kb_id: str) -> KnowledgeBaseProfile | None:
    return await db.get(KnowledgeBaseProfile, kb_id)


async def _ensure_profile(
    db: AsyncSession,
    kb_id: str,
    *,
    domain: str = "general",
    visibility: str = "internal",
    language: str = "zh-CN",
    tags: list[str] | None = None,
    allowed_roles: list[str] | None = None,
) -> KnowledgeBaseProfile:
    profile = await _get_profile(db, kb_id)
    if profile:
        return profile
    profile = KnowledgeBaseProfile(
        knowledge_base_id=kb_id,
        domain=domain,
        visibility=visibility,
        language=language,
        tags_json=json.dumps(tags or [], ensure_ascii=False),
        allowed_roles_json=json.dumps(allowed_roles or [], ensure_ascii=False),
    )
    db.add(profile)
    await db.flush()
    return profile


def _to_response(kb: KnowledgeBase, profile: KnowledgeBaseProfile | None) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        domain=profile.domain if profile else "general",
        visibility=profile.visibility if profile else "internal",
        language=profile.language if profile else "zh-CN",
        tags=loads_str_list(profile.tags_json if profile else None),
        allowed_roles=loads_str_list(profile.allowed_roles_json if profile else None),
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
    profile = await _ensure_profile(
        db,
        kb.id,
        domain=body.domain,
        visibility=body.visibility,
        language=body.language,
        tags=body.tags,
        allowed_roles=body.allowed_roles,
    )
    return APIResponse(data=_to_response(kb, profile))


@router.get("", response_model=APIResponse[PageResult[KnowledgeBaseResponse]])
async def list_knowledge_bases(
    page: int = 1,
    page_size: int = 20,
    domain: str = "",
    visibility: str = "",
    language: str = "",
    tag: str = "",
    role: str = "",
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    kb_result = await db.execute(
        select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
    )
    knowledge_bases = list(kb_result.scalars().all())
    profile_result = await db.execute(select(KnowledgeBaseProfile))
    profiles = {item.knowledge_base_id: item for item in profile_result.scalars().all()}

    filtered = [
        kb for kb in knowledge_bases
        if (
            not profiles.get(kb.id) and not any([domain, visibility, language, tag, role])
        ) or (
            profiles.get(kb.id) and profile_matches(
                profile_domain=profiles[kb.id].domain,
                profile_visibility=profiles[kb.id].visibility,
                profile_language=profiles[kb.id].language,
                tags_json=profiles[kb.id].tags_json,
                allowed_roles_json=profiles[kb.id].allowed_roles_json,
                domain=domain,
                visibility=visibility,
                language=language,
                tag=tag,
                role=role,
            )
        )
    ]
    total = len(filtered)
    page_items = filtered[offset: offset + page_size]
    items = [_to_response(kb, profiles.get(kb.id)) for kb in page_items]

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
    profile = await _get_profile(db, kb_id)
    return APIResponse(data=_to_response(kb, profile))


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

    profile = await _ensure_profile(db, kb_id)
    if body.domain is not None:
        profile.domain = body.domain
    if body.visibility is not None:
        profile.visibility = body.visibility
    if body.language is not None:
        profile.language = body.language
    if body.tags is not None:
        profile.tags_json = json.dumps(body.tags, ensure_ascii=False)
    if body.allowed_roles is not None:
        profile.allowed_roles_json = json.dumps(body.allowed_roles, ensure_ascii=False)
    profile.updated_at = datetime.now()
    await db.flush()

    return APIResponse(data=_to_response(kb, profile))


@router.delete("/{kb_id}", response_model=APIResponse)
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    profile = await _get_profile(db, kb_id)
    if profile:
        await db.delete(profile)
    await db.delete(kb)
    return APIResponse(message="已删除")
