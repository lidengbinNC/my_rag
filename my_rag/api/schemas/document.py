from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str
    filename: str
    file_size: int = Field(description="文件大小（字节）")
    content_type: str | None = None
    status: str = Field(description="处理状态: pending / processing / completed / failed")
    chunk_count: int = 0
    knowledge_base_id: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    items: list[DocumentResponse] = Field(default_factory=list)
    total: int = 0
