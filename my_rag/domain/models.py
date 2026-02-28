"""
领域数据模型

与 ORM 模型分离，表示业务流转中的数据结构。
面试考点：贫血模型 vs 充血模型、dataclass vs Pydantic
"""

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """文档解析结果"""
    content: str
    metadata: dict = field(default_factory=dict)
    pages: list[str] | None = None


@dataclass
class TextChunk:
    """文本分块结果"""
    content: str
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)
