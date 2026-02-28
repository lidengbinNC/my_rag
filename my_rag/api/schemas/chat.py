from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    knowledge_base_id: str = Field(..., description="知识库 ID")
    conversation_id: str | None = Field(default=None, description="对话 ID（多轮对话）")
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数")
    stream: bool = Field(default=True, description="是否流式返回")


class SourceDocument(BaseModel):
    """引用的源文档片段"""
    content: str = Field(..., description="文档内容片段")
    source: str = Field(..., description="来源文件名")
    score: float = Field(default=0.0, description="相关性分数")
    chunk_id: str | None = Field(default=None, description="分块 ID")


class TokenUsage(BaseModel):
    """Token 使用量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(..., description="回答内容")
    sources: list[SourceDocument] = Field(default_factory=list, description="引用来源")
    conversation_id: str = Field(..., description="对话 ID")
    usage: TokenUsage | None = Field(default=None, description="Token 用量")


class MessageResponse(BaseModel):
    """消息记录"""
    id: str
    role: str
    content: str
    sources: list[SourceDocument] | None = None
    created_at: datetime


class ConversationResponse(BaseModel):
    """对话记录"""
    id: str
    title: str | None = None
    knowledge_base_id: str
    message_count: int = 0
    created_at: datetime
    updated_at: datetime
