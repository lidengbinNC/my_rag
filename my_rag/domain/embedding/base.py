"""
Embedding 抽象基类

面试考点：
- Embedding 是将文本映射到高维向量空间，使语义相近的文本向量距离更近
- embed_query vs embed_documents：部分模型（如 BGE）对查询和文档使用不同的 instruction prefix
- 批处理优化：避免逐条请求，减少 I/O 开销和模型推理次数
- Hybrid Retrieval：同一模型可同时输出 Dense + Sparse 表示，用于 Milvus 原生混合检索
"""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...

    @property
    def supports_sparse(self) -> bool:
        """当前 Embedding 是否支持输出稀疏向量。"""
        return False

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量将文档文本转为向量"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """将查询文本转为向量（可能使用不同的 instruction）"""
        ...

    async def embed_documents_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """批量返回 dense + sparse 表示；默认实现要求子类显式覆写。"""
        raise NotImplementedError(f"{type(self).__name__} does not support sparse embeddings")

    async def embed_query_hybrid(self, text: str) -> tuple[list[float], dict[int, float]]:
        """返回单条查询的 dense + sparse 表示；默认实现要求子类显式覆写。"""
        raise NotImplementedError(f"{type(self).__name__} does not support sparse embeddings")
