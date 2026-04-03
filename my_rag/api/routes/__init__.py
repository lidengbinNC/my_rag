from fastapi import APIRouter

from my_rag.api.routes import (
    chat,
    document,
    evaluation,
    health,
    knowledge,
    metrics,
    notification,
    search,
    websocket,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["系统"])
api_router.include_router(knowledge.router, tags=["知识库"])
api_router.include_router(document.router, tags=["文档"])
api_router.include_router(search.router, tags=["检索"])
api_router.include_router(chat.router, tags=["对话"])
api_router.include_router(websocket.router, tags=["WebSocket"])
api_router.include_router(notification.router, tags=["通知"])
api_router.include_router(evaluation.router, tags=["评估"])
api_router.include_router(metrics.router, tags=["监控"])
