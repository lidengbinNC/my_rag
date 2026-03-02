"""
Embedding 抽象基类

面试考点：
- Embedding 是将文本映射到高维向量空间，使语义相近的文本向量距离更近
- embed_query vs embed_documents：部分模型（如 BGE）对查询和文档使用不同的 instruction prefix
- 批处理优化：避免逐条请求，减少 I/O 开销和模型推理次数
"""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量将文档文本转为向量"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """将查询文本转为向量（可能使用不同的 instruction）"""
        ...
