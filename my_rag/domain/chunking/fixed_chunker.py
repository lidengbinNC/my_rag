"""
固定大小分块器

面试考点：
- 最简单的分块基线方案
- 基于 Token 分割而非字符分割（与 LLM 对齐）
- 滑动窗口重叠策略保证上下文衔接
"""

from my_rag.domain.chunking.base import BaseChunker
from my_rag.domain.models import TextChunk
from my_rag.utils.token_counter import encode_tokens, decode_tokens


class FixedChunker(BaseChunker):
    """按固定 Token 数分块，带滑动窗口重叠"""

    def chunk(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        meta = metadata or {}
        tokens = encode_tokens(text)

        if len(tokens) <= self.chunk_size:
            return [
                TextChunk(
                    content=text.strip(),
                    chunk_index=0,
                    token_count=len(tokens),
                    metadata={**meta, "chunker": "fixed"},
                )
            ]

        chunks: list[TextChunk] = []
        start = 0
        idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = decode_tokens(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        chunk_index=idx,
                        token_count=len(chunk_tokens),
                        metadata={**meta, "chunker": "fixed"},
                    )
                )
                idx += 1

            if end >= len(tokens):
                break
            start += step

        return chunks
