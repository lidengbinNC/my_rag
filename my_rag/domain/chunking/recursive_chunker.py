"""
递归字符分块器

面试考点：
- 这是 LangChain RecursiveCharacterTextSplitter 的核心算法
- 按分隔符层级递归分割：段落 → 句子 → 词，尽可能保留语义完整性
- 先切大块，再合并小块到 chunk_size 以内
- 中文分隔符优先级需要特别处理
"""

from my_rag.domain.chunking.base import BaseChunker
from my_rag.domain.models import TextChunk
from my_rag.utils.token_counter import count_tokens


DEFAULT_SEPARATORS = [
    "\n\n",     # 段落
    "\n",       # 换行
    "。",       # 中文句号
    ".",        # 英文句号
    "！", "!",  # 感叹号
    "？", "?",  # 问号
    "；", ";",  # 分号
    " ",        # 空格
    "",         # 逐字符（最后兜底）
]


class RecursiveChunker(BaseChunker):
    """递归字符文本分块器 — RAG 领域最常用的分块策略"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        meta = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._merge_chunks(raw_chunks)

        results: list[TextChunk] = []
        for idx, chunk_text in enumerate(merged):
            token_count = count_tokens(chunk_text)
            results.append(
                TextChunk(
                    content=chunk_text,
                    chunk_index=idx,
                    token_count=token_count,
                    metadata={**meta, "chunker": "recursive"},
                )
            )
        return results

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """递归分割核心逻辑"""
        if not text.strip():
            return []

        if count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if separator == "":
            return self._split_by_chars(text)

        splits = text.split(separator)
        splits = [s for s in splits if s.strip()]

        good_chunks: list[str] = []
        current = ""

        for piece in splits:
            candidate = (current + separator + piece) if current else piece

            if count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_chunks.append(current.strip())

                if count_tokens(piece) <= self.chunk_size:
                    current = piece
                else:
                    sub_chunks = self._recursive_split(piece, remaining_seps)
                    good_chunks.extend(sub_chunks)
                    current = ""

        if current.strip():
            good_chunks.append(current.strip())

        return good_chunks

    def _split_by_chars(self, text: str) -> list[str]:
        """按字符切割（最终兜底策略）"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size * 4
            chunk = text[start:end]

            while count_tokens(chunk) > self.chunk_size and len(chunk) > 1:
                chunk = chunk[: len(chunk) * 3 // 4]

            if chunk.strip():
                chunks.append(chunk.strip())
            start += max(len(chunk) - self.chunk_overlap * 4, 1)
        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """合并过小的块，并添加重叠"""
        if not chunks:
            return []

        merged: list[str] = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            candidate = current + "\n" + chunks[i]

            if count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current.strip())

                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current)
                    current = overlap_text + "\n" + chunks[i] if overlap_text else chunks[i]
                else:
                    current = chunks[i]

        if current.strip():
            merged.append(current.strip())

        return merged

    def _get_overlap_text(self, text: str) -> str:
        """从文本末尾提取重叠部分"""
        if self.chunk_overlap <= 0:
            return ""

        sentences = text.replace("。", "。\n").replace(".", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]

        overlap = ""
        for s in reversed(sentences):
            candidate = s + " " + overlap if overlap else s
            if count_tokens(candidate) <= self.chunk_overlap:
                overlap = candidate
            else:
                break
        return overlap.strip()
