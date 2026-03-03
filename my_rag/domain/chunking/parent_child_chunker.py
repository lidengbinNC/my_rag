"""
Parent-Child 分块策略

面试考点（高频必考）：
- 问题：小块精确但缺上下文，大块完整但噪声多 → 两难
- Parent-Child 思想：用小块（child）做检索保证精度，返回大块（parent）保证上下文完整
- 实现方式：
  1. 先按大窗口切出 parent chunks
  2. 对每个 parent 再按小窗口切出 child chunks
  3. 检索时匹配 child，但返回给 LLM 的是对应的 parent
- 优势：兼顾检索精度与上下文完整性
- 对比 LangChain 的 ParentDocumentRetriever

本实现：
- parent_chunk_size: 大块尺寸（如 1024 tokens）
- child_chunk_size:  小块尺寸（如 256 tokens）
- 返回的 TextChunk 中通过 metadata["parent_content"] 携带 parent 原文
"""

from my_rag.domain.chunking.base import BaseChunker
from my_rag.domain.chunking.recursive_chunker import RecursiveChunker
from my_rag.domain.models import TextChunk


class ParentChildChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self._parent_chunker = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self._child_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        metadata = metadata or {}
        parent_chunks = self._parent_chunker.chunk(text, metadata=metadata)

        all_children: list[TextChunk] = []
        global_idx = 0

        for parent in parent_chunks:
            children = self._child_chunker.chunk(parent.content)

            for child in children:
                child_meta = {
                    **metadata,
                    "parent_index": parent.chunk_index,
                    "parent_content": parent.content,
                    "chunking_strategy": "parent_child",
                }
                all_children.append(TextChunk(
                    content=child.content,
                    chunk_index=global_idx,
                    token_count=child.token_count,
                    metadata=child_meta,
                ))
                global_idx += 1

        return all_children
