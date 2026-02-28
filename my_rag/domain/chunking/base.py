"""
文本分块器抽象基类

面试考点：
- 分块大小对 RAG 效果的影响：小块精确但缺上下文，大块完整但噪声多
- chunk_overlap 的作用：保证语义连续性，避免在句子中间断裂
- Token 计数 vs 字符计数：LLM 以 Token 为单位，必须用 Token 衡量
"""

from abc import ABC, abstractmethod

from my_rag.domain.models import TextChunk
from my_rag.utils.token_counter import count_tokens


class BaseChunker(ABC):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        """将文本分割为多个块"""
        ...

    @property
    def strategy_name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def count_tokens(text: str) -> int:
        return count_tokens(text)
