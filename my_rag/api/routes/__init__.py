from fastapi import APIRouter

from my_rag.api.routes import health, knowledge, document, chat

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["系统"])
api_router.include_router(knowledge.router, tags=["知识库"])
api_router.include_router(document.router, tags=["文档"])
api_router.include_router(chat.router, tags=["对话"])
