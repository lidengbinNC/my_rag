"""
对话接口 - 接入真实 RAG Pipeline

面试考点：
- SSE（Server-Sent Events）流式响应：HTTP 长连接，服务端单向推送
- AsyncGenerator 实现 token-by-token 流式输出
- 多轮对话上下文管理
- RAG Pipeline 集成：Retrieve → Prompt → Generate
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
from my_rag.core.dependencies import get_dingtalk_notifier, get_rag_pipeline
from my_rag.infrastructure.database import (
    Conversation,
    KnowledgeBase,
    Message,
    get_db,
)
from my_rag.utils.logger import get_logger

router = APIRouter(prefix="/chat")
logger = get_logger(__name__)


async def _notify_dingtalk(query: str, answer: str) -> None:
    """问答完成后异步推送钉钉通知（旁路逻辑，失败不影响主流程）"""
    notifier = get_dingtalk_notifier()
    if notifier is None:
        return
    try:
        md = (
            f"### RAG 问答通知\n\n"
            f"**问题：** {query}\n\n"
            f"**回答：** {answer[:500]}{'...' if len(answer) > 500 else ''}\n\n"
            f"---\n"
            f"*来自 MyRAG 系统*"
        )
        await notifier.send_markdown(title="RAG 问答通知", text=md)
    except Exception as exc:
        logger.warning("dingtalk_notify_failed", error=str(exc))


def _build_chat_history(messages: list[Message], max_turns: int = 5) -> str:
    """构建对话历史字符串（最近 N 轮）"""
    recent = messages[-(max_turns * 2):]
    parts = []
    for msg in recent:
        role_label = "用户" if msg.role == "user" else "助手"
        parts.append(f"{role_label}: {msg.content}")
    return "\n".join(parts)


async def _stream_rag_response(query: str, knowledge_base_id: str, conversation_id: str, chat_history: str):
    """SSE 流式 RAG 响应"""
    pipeline = get_rag_pipeline()

    async for event in pipeline.stream(
        query=query,
        knowledge_base_id=knowledge_base_id,
        chat_history=chat_history,
    ):
        if event["type"] == "retrieval":
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        elif event["type"] == "token":
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        elif event["type"] == "done":
            done_event = {
                "type": "done",
                "conversation_id": conversation_id,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
            asyncio.create_task(_notify_dingtalk(query, event.get("full_answer", "")))


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

    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    history_messages = list(history_result.scalars().all())
    chat_history = _build_chat_history(history_messages)

    if body.stream:
        await db.commit()
        return StreamingResponse(
            _stream_rag_response(body.query, body.knowledge_base_id, conversation.id, chat_history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conversation.id,
            },
        )

    pipeline = get_rag_pipeline()
    rag_result = await pipeline.run(
        query=body.query,
        knowledge_base_id=body.knowledge_base_id,
        chat_history=chat_history,
    )

    sources = [
        SourceDocument(
            content=s.content,
            source=s.source,
            score=s.score,
            chunk_id=s.chunk_id,
        )
        for s in rag_result.sources
    ]
    sources_json = json.dumps([s.model_dump() for s in sources], ensure_ascii=False)

    assistant_msg = Message(
        id=str(uuid.uuid4()),
        role="assistant",
        content=rag_result.answer,
        sources_json=sources_json,
        conversation_id=conversation.id,
        created_at=datetime.now(),
    )
    db.add(assistant_msg)

    asyncio.create_task(_notify_dingtalk(body.query, rag_result.answer))

    return APIResponse(
        data=ChatResponse(
            answer=rag_result.answer,
            sources=sources,
            conversation_id=conversation.id,
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
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
