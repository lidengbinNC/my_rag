"""
向量存储抽象基类

面试考点：
- 向量数据库的核心操作：add / search / delete
- 相似度度量：cosine / L2 / inner product
- 元数据过滤：在向量检索基础上施加标量条件过滤
- Hybrid Search：Dense + Sparse 统一写入与混合召回
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

    @property
    def supports_hybrid(self) -> bool:
        return False

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

    async def add_hybrid(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        sparse_embeddings: list[dict[int, float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid upsert")

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        """向量相似度检索"""
        ...

    async def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse_embedding: dict[int, float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        ranker: str = "weighted",
        candidate_limit: int | None = None,
        rrf_k: int = 60,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid search")

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
