"""
BM25 稀疏检索器

面试考点：
- BM25 是基于词频 (TF) 和逆文档频率 (IDF) 的经典检索算法
- 公式核心：score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1 * (1 - b + b * |d|/avgdl))
  - k1 控制词频饱和度（默认 1.5），b 控制文档长度归一化（默认 0.75）
- 优势：对精确关键词匹配、专业术语、产品型号等效果好
- 劣势：无法理解同义词、语义相似的表达
- 与稠密检索形成互补，是混合检索的基础
"""

import asyncio
import jieba
from rank_bm25 import BM25Okapi

from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


class SparseRetriever(BaseRetriever):

    def __init__(self):
        self._corpus_chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, chunks: list[dict]) -> None:
        """
        构建 BM25 索引
        chunks: [{"id": str, "content": str, "source": str, "knowledge_base_id": str, ...}]
        """
        self._corpus_chunks = chunks
        self._tokenized_corpus = [_tokenize(c["content"]) for c in chunks]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("bm25_index_built", corpus_size=len(chunks))

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        if not self._bm25 or not self._corpus_chunks:
            return []

        tokenized_query = await asyncio.to_thread(_tokenize, query)
        scores = await asyncio.to_thread(self._bm25.get_scores, tokenized_query)

        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if knowledge_base_id and self._corpus_chunks[idx].get("knowledge_base_id") != knowledge_base_id:
                continue
            candidates.append((idx, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for idx, score in candidates[:top_k]:
            chunk = self._corpus_chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk["id"],
                content=chunk["content"],
                score=score,
                source=chunk.get("source", ""),
                metadata=chunk,
            ))

        return results
