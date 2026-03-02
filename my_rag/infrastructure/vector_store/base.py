"""
向量存储抽象基类

面试考点：
- 向量数据库的核心操作：add / search / delete
- 相似度度量：cosine / L2 / inner product
- 元数据过滤：在向量检索基础上施加标量条件过滤
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VectorSearchResult:
    """向量检索结果"""
    chunk_id: str
    score: float
    content: str = ""
    metadata: dict = field(default_factory=dict)


class BaseVectorStore(ABC):

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """批量添加向量"""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        """向量相似度检索"""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """按 ID 删除向量"""
        ...

    @abstractmethod
    async def delete_by_metadata(self, key: str, value: str) -> None:
        """按元数据条件删除"""
        ...

    @abstractmethod
    def count(self) -> int:
        """向量总数"""
        ...
