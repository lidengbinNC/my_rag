from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """检索请求（仅返回知识片段，不生成最终答案）。"""

    query: str = Field(..., min_length=1, max_length=2000, description="检索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    knowledge_base_id: str | None = Field(default=None, description="知识库 ID")
    knowledge_base: str | None = Field(
        default=None,
        description="知识库名称或 ID，兼容旧集成方式",
    )
    domain: str | None = Field(default=None, description="知识域，如 faq/sop/policy/ticket")
    agent_role: str | None = Field(default=None, description="访问角色，如 bot/agent/admin")
    language: str | None = Field(default=None, description="语言")
    tag: str | None = Field(default=None, description="标签过滤")


class SearchResult(BaseModel):
    """单条检索结果。"""

    content: str = Field(..., description="知识片段内容")
    score: float = Field(default=0.0, description="相关性分数")
    source: str = Field(default="", description="来源文件名")
    chunk_id: str | None = Field(default=None, description="分块 ID")
    metadata: dict = Field(default_factory=dict, description="附加元数据")


class SearchResponse(BaseModel):
    """检索响应契约。"""

    results: list[SearchResult] = Field(default_factory=list)
    total: int = Field(default=0)
    query: str = Field(default="")
    knowledge_base_id: str = Field(default="")
    knowledge_base_name: str = Field(default="")
    domain: str = Field(default="")
    retrieval_mode: str = Field(default="retrieval_only")
