"""检索器抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    source: str = ""
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        ...
