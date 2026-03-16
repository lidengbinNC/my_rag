"""
BM25 稀疏检索器

面试考点：
- BM25 是基于词频 (TF) 和逆文档频率 (IDF) 的经典检索算法
- 公式核心：score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1 * (1 - b + b * |d|/avgdl))
  - k1 控制词频饱和度（默认 1.5），b 控制文档长度归一化（默认 0.75）
- 优势：对精确关键词匹配、专业术语、产品型号等效果好
- 劣势：无法理解同义词、语义相似的表达
- 与稠密检索形成互补，是混合检索的基础

索引更新策略（面试考点）：
- 全量重建（build_index）：启动时从 DB 加载全部 chunk，适合冷启动/重启恢复
- 增量追加（add_chunks）：新文档处理完成后只追加新 chunk，无需查 DB，O(新增) 而非 O(全量)
- rank_bm25 的 BM25Okapi 不支持增量更新，追加后需重新构造对象（但只需对新增部分分词）
"""

import asyncio
import jieba
from rank_bm25 import BM25Okapi

from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


def _tokenize_batch(texts: list[str]) -> list[list[str]]:
    return [_tokenize(t) for t in texts]


class SparseRetriever(BaseRetriever):

    def __init__(self):
        self._corpus_chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, chunks: list[dict]) -> None:
        """
        全量重建 BM25 索引（冷启动 / 删除文档后使用）

        chunks: [{"id": str, "content": str, "source": str, "knowledge_base_id": str}]

        面试考点：
        - 全量重建时间复杂度 O(N * avg_tokens)，N 为语料库大小
        - 适合：应用启动时从 DB 恢复索引、删除文档后需要收缩索引
        - 不适合：每次新增文档都调用（文档多时性能差）
        """
        self._corpus_chunks = chunks
        self._tokenized_corpus = [_tokenize(c["content"]) for c in chunks]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("bm25_index_built", corpus_size=len(chunks))

    def add_chunks(self, new_chunks: list[dict]) -> None:
        """
        增量追加新 chunk（新文档处理完成后调用）

        面试考点：
        - rank_bm25 不支持真正的增量更新（IDF 需要全局统计），追加后必须重建 BM25Okapi 对象
        - 但只需对新增部分分词，旧部分的分词结果已缓存在 _tokenized_corpus 中
        - 时间复杂度 O(new_chunks * avg_tokens + N)，N 为重建 BM25 对象的开销
        - 相比全量重建（需要重新查 DB + 对全部 chunk 分词），增量追加节省了 DB 查询和旧 chunk 分词

        Args:
            new_chunks: 新增的 chunk 列表，格式同 build_index
        """
        if not new_chunks:
            return

        new_tokenized = [_tokenize(c["content"]) for c in new_chunks]

        self._corpus_chunks.extend(new_chunks)
        self._tokenized_corpus.extend(new_tokenized)

        # 重建 BM25 对象（必须，因为 IDF 依赖全局文档频率）
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "bm25_index_updated",
            added=len(new_chunks),
            total=len(self._corpus_chunks),
        )

    def remove_by_document_id(self, document_id: str) -> int:
        """
        按 document_id 从索引中删除对应 chunk（删除文档时调用）

        面试考点：
        - BM25 索引是内存结构，删除后需重建 BM25Okapi 对象
        - 删除操作后索引收缩，IDF 重新计算，结果更准确

        Returns:
            删除的 chunk 数量
        """
        before = len(self._corpus_chunks)
        keep_indices = [
            i for i, c in enumerate(self._corpus_chunks)
            if c.get("document_id") != document_id
        ]

        if len(keep_indices) == before:
            return 0  # 没有匹配的 chunk，无需重建

        self._corpus_chunks = [self._corpus_chunks[i] for i in keep_indices]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

        removed = before - len(self._corpus_chunks)
        logger.info(
            "bm25_index_chunks_removed",
            document_id=document_id,
            removed=removed,
            remaining=len(self._corpus_chunks),
        )
        return removed

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
