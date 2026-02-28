"""
对话接口 - Phase 1 使用 Mock 响应，后续接入真实 RAG Pipeline

面试考点：
- SSE（Server-Sent Events）流式响应
- AsyncGenerator 实现流式输出
- 多轮对话上下文管理
"""

import asyncio
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from my_rag.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    SourceDocument,
    TokenUsage,
)
from my_rag.api.schemas.common import APIResponse
from my_rag.infrastructure.database import (
    Conversation,
    KnowledgeBase,
    Message,
    get_db,
)

router = APIRouter(prefix="/chat")

MOCK_SOURCES = [
    SourceDocument(
        content="RAG（检索增强生成）是一种结合信息检索与语言模型生成的技术框架...",
        source="rag_overview.pdf",
        score=0.95,
        chunk_id="chunk-001",
    ),
    SourceDocument(
        content="向量数据库通过 ANN（近似最近邻）算法实现高效的向量相似度检索...",
        source="vector_db_guide.md",
        score=0.87,
        chunk_id="chunk-002",
    ),
]


def _mock_answer(query: str) -> str:
    """Phase 1 模拟回答"""
    return (
        f"感谢您的提问！关于「{query}」，"
        "根据知识库中的文档，我为您整理了以下信息：\n\n"
        "**核心要点：**\n\n"
        "1. RAG（Retrieval-Augmented Generation）是一种将信息检索与大语言模型"
        "生成能力相结合的技术框架，能够有效减少幻觉问题。\n\n"
        "2. 系统通过将用户查询转换为向量表示，在向量数据库中检索最相关的文档片段，"
        "然后将这些片段作为上下文传递给 LLM 进行回答生成。\n\n"
        "3. 混合检索（Hybrid Retrieval）结合了稠密向量检索和稀疏关键词检索（BM25），"
        "通过 RRF（Reciprocal Rank Fusion）算法融合排序结果，显著提升检索质量。\n\n"
        "> **注意**：当前为演示模式，回答基于模拟数据。接入真实 RAG Pipeline 后，"
        "将根据您上传的文档内容进行精确回答。"
    )


async def _stream_mock_response(query: str, conversation_id: str):
    """SSE 流式模拟响应"""
    retrieval_event = {
        "type": "retrieval",
        "documents": [s.model_dump() for s in MOCK_SOURCES],
    }
    yield f"data: {json.dumps(retrieval_event, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0.3)

    answer = _mock_answer(query)
    for char in answer:
        token_event = {"type": "token", "content": char}
        yield f"data: {json.dumps(token_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.02)

    done_event = {
        "type": "done",
        "conversation_id": conversation_id,
        "usage": {"prompt_tokens": 520, "completion_tokens": 180, "total_tokens": 700},
    }
    yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"


@router.post("/completions")
async def chat_completions(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, body.knowledge_base_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if body.conversation_id:
        conversation = await db.get(Conversation, body.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
    else:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=body.query[:50],
            knowledge_base_id=body.knowledge_base_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(conversation)
        await db.flush()

    user_msg = Message(
        id=str(uuid.uuid4()),
        role="user",
        content=body.query,
        conversation_id=conversation.id,
        created_at=datetime.now(),
    )
    db.add(user_msg)

    if body.stream:
        await db.commit()
        return StreamingResponse(
            _stream_mock_response(body.query, conversation.id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conversation.id,
            },
        )

    answer = _mock_answer(body.query)
    sources_json = json.dumps([s.model_dump() for s in MOCK_SOURCES], ensure_ascii=False)

    assistant_msg = Message(
        id=str(uuid.uuid4()),
        role="assistant",
        content=answer,
        sources_json=sources_json,
        conversation_id=conversation.id,
        created_at=datetime.now(),
    )
    db.add(assistant_msg)

    return APIResponse(
        data=ChatResponse(
            answer=answer,
            sources=MOCK_SOURCES,
            conversation_id=conversation.id,
            usage=TokenUsage(prompt_tokens=520, completion_tokens=180, total_tokens=700),
        )
    )


@router.get("/conversations", response_model=APIResponse[list[ConversationResponse]])
async def list_conversations(
    knowledge_base_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Conversation).order_by(Conversation.updated_at.desc())
    if knowledge_base_id:
        query = query.where(Conversation.knowledge_base_id == knowledge_base_id)

    result = await db.execute(query)
    conversations = result.scalars().all()

    items = []
    for conv in conversations:
        msg_result = await db.execute(
            select(Message).where(Message.conversation_id == conv.id)
        )
        msg_count = len(msg_result.scalars().all())
        items.append(
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                knowledge_base_id=conv.knowledge_base_id,
                message_count=msg_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
        )

    return APIResponse(data=items)


@router.delete("/conversations/{conv_id}", response_model=APIResponse)
async def delete_conversation(
    conv_id: str,
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    await db.delete(conv)
    return APIResponse(message="对话已删除")
