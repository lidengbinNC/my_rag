from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field(default="", max_length=500, description="描述")
    domain: str = Field(default="general", max_length=64, description="知识域")
    visibility: str = Field(default="internal", max_length=32, description="可见范围")
    language: str = Field(default="zh-CN", max_length=32, description="语言")
    tags: list[str] = Field(default_factory=list, description="标签")
    allowed_roles: list[str] = Field(default_factory=list, description="允许访问角色")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    domain: str | None = Field(default=None, max_length=64)
    visibility: str | None = Field(default=None, max_length=32)
    language: str | None = Field(default=None, max_length=32)
    tags: list[str] | None = Field(default=None)
    allowed_roles: list[str] | None = Field(default=None)


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    id: str
    name: str
    description: str = ""
    domain: str = "general"
    visibility: str = "internal"
    language: str = "zh-CN"
    tags: list[str] = Field(default_factory=list)
    allowed_roles: list[str] = Field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime
