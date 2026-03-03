"""
WebSocket 实时对话接口

面试考点：
- WebSocket vs SSE：
  - SSE：单向（服务端→客户端），基于 HTTP，自动重连，简单
  - WebSocket：双向全双工，独立协议（ws://），需手动维护心跳和重连
- WebSocket 适合场景：实时聊天、协同编辑、游戏
- 在 FastAPI 中 WebSocket 的生命周期管理
- 连接状态管理：connect / disconnect / error handling
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from my_rag.core.dependencies import get_rag_pipeline
from my_rag.infrastructure.database import (
    Conversation,
    KnowledgeBase,
    Message,
    async_session_factory,
)
from my_rag.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info("ws_connected", client_id=client_id, total=len(self.active_connections))

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        logger.info("ws_disconnected", client_id=client_id, total=len(self.active_connections))

    async def send_json(self, client_id: str, data: dict):
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_message(websocket, client_id, data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error("ws_error", client_id=client_id, error=str(e))
        manager.disconnect(client_id)


async def _handle_message(websocket: WebSocket, client_id: str, data: dict):
    msg_type = data.get("type", "chat")

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})
        return

    if msg_type != "chat":
        await websocket.send_json({"type": "error", "message": f"未知消息类型: {msg_type}"})
        return

    query = data.get("query", "").strip()
    kb_id = data.get("knowledge_base_id", "")
    conversation_id = data.get("conversation_id")

    if not query or not kb_id:
        await websocket.send_json({"type": "error", "message": "query 和 knowledge_base_id 为必填"})
        return

    async with async_session_factory() as session:
        kb = await session.get(KnowledgeBase, kb_id)
        if not kb:
            await websocket.send_json({"type": "error", "message": "知识库不存在"})
            return

        if conversation_id:
            conversation = await session.get(Conversation, conversation_id)
        else:
            conversation = Conversation(
                id=str(uuid.uuid4()),
                title=query[:50],
                knowledge_base_id=kb_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(conversation)
            await session.flush()

        user_msg = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=query,
            conversation_id=conversation.id,
            created_at=datetime.now(),
        )
        session.add(user_msg)

        history_result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        )
        history_messages = list(history_result.scalars().all())
        chat_history = _build_ws_history(history_messages)

        await session.commit()

    pipeline = get_rag_pipeline()
    full_answer = ""

    async for event in pipeline.stream(
        query=query,
        knowledge_base_id=kb_id,
        chat_history=chat_history,
    ):
        if event["type"] == "retrieval":
            await websocket.send_json(event)
        elif event["type"] == "token":
            full_answer += event["content"]
            await websocket.send_json(event)
        elif event["type"] == "done":
            await websocket.send_json({
                "type": "done",
                "conversation_id": conversation.id,
                "full_answer": full_answer,
            })

    async with async_session_factory() as session:
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=full_answer,
            conversation_id=conversation.id,
            created_at=datetime.now(),
        )
        session.add(assistant_msg)
        await session.commit()


def _build_ws_history(messages: list[Message], max_turns: int = 5) -> str:
    recent = messages[-(max_turns * 2):]
    parts = []
    for msg in recent:
        role_label = "用户" if msg.role == "user" else "助手"
        parts.append(f"{role_label}: {msg.content}")
    return "\n".join(parts)
