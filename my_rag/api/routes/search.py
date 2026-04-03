"""面向外部 Agent 的 retrieval-only 检索接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.search import SearchRequest, SearchResponse, SearchResult
from my_rag.config.settings import settings
from my_rag.core.dependencies import get_rag_pipeline
from my_rag.domain.knowledge import loads_str_list
from my_rag.infrastructure.database import KnowledgeBase, KnowledgeBaseProfile, get_db
from my_rag.utils.logger import get_logger

router = APIRouter(prefix="/search")
logger = get_logger(__name__)


def _verify_agent_key(x_api_key: str | None) -> None:
    expected = settings.integration.agent_api_key.strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")


async def _resolve_knowledge_base(
    db: AsyncSession,
    body: SearchRequest,
) -> tuple[KnowledgeBase, KnowledgeBaseProfile | None]:
    kb_ref = (body.knowledge_base_id or body.knowledge_base or "").strip()
    if not kb_ref:
        domain = (body.domain or "").strip()
        if not domain:
            raise HTTPException(status_code=400, detail="knowledge_base_id / knowledge_base / domain 至少提供一个")
        result = await db.execute(select(KnowledgeBaseProfile).where(KnowledgeBaseProfile.domain == domain))
        profiles = list(result.scalars().all())
        if body.language:
            profiles = [item for item in profiles if item.language == body.language]
        if body.tag:
            profiles = [
                item for item in profiles
                if body.tag in loads_str_list(item.tags_json)
            ]
        if body.agent_role:
            filtered = []
            for item in profiles:
                roles = loads_str_list(item.allowed_roles_json)
                if not roles or body.agent_role in roles:
                    filtered.append(item)
            profiles = filtered
        if not profiles:
            raise HTTPException(status_code=404, detail="未找到匹配知识域的知识库")
        profile = sorted(profiles, key=lambda item: item.updated_at, reverse=True)[0]
        kb = await db.get(KnowledgeBase, profile.knowledge_base_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return kb, profile

    kb = await db.get(KnowledgeBase, kb_ref)
    if kb:
        return kb, await db.get(KnowledgeBaseProfile, kb.id)

    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.name == kb_ref).limit(1)
    )
    kb = result.scalar_one_or_none()
    if kb:
        return kb, await db.get(KnowledgeBaseProfile, kb.id)

    raise HTTPException(status_code=404, detail="知识库不存在")


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
    x_api_key: str | None = Header(default=None),
):
    """只做检索，不生成最终答案。"""
    _verify_agent_key(x_api_key)
    kb, profile = await _resolve_knowledge_base(db, body)

    pipeline = get_rag_pipeline()
    results = await pipeline.search(
        query=body.query,
        knowledge_base_id=kb.id,
        top_k=body.top_k,
    )

    max_chars = settings.integration.search_result_max_chars
    items = [
        SearchResult(
            content=result.content[:max_chars],
            score=result.score,
            source=result.source,
            chunk_id=result.chunk_id,
            metadata=result.metadata,
        )
        for result in results
    ]
    logger.info(
        "agent_search_completed",
        knowledge_base_id=kb.id,
        top_k=body.top_k,
        result_count=len(items),
    )
    return SearchResponse(
        results=items,
        total=len(items),
        query=body.query,
        knowledge_base_id=kb.id,
        knowledge_base_name=kb.name,
        domain=profile.domain if profile else "",
    )
