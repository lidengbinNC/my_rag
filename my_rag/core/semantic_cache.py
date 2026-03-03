"""
语义缓存

面试考点：
- 传统缓存（Redis）：基于精确 key 匹配，"什么是RAG" 和 "RAG是什么" 被视为不同 key
- 语义缓存：将 query embedding 后做向量相似度匹配，语义相近的问题命中同一条缓存
- 实现：独立的 FAISS 索引存 query embedding → 缓存条目映射
- 阈值控制：similarity > threshold 才视为命中，避免误命中
- 缓存淘汰：LRU + TTL，控制内存占用
- 适用场景：高频重复问题的企业知识库，可大幅降低 LLM 调用成本
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    created_at: float = 0.0


class SemanticCache:
    """基于向量相似度的语义缓存"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        similarity_threshold: float = 0.92,
        max_size: int = 500,
        ttl_seconds: int = 3600,
    ):
        self._embedding = embedding
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._ttl = ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: list[list[float]] = []
        self._keys: list[str] = []

    async def get(self, query: str) -> CacheEntry | None:
        """查询缓存：embedding → cosine similarity → 阈值判断"""
        if not self._cache:
            return None

        self._evict_expired()

        if not self._cache:
            return None

        query_emb = await self._embedding.embed_query(query)
        query_vec = np.array([query_emb], dtype=np.float32)
        cache_matrix = np.array(self._embeddings, dtype=np.float32)

        similarities = np.dot(cache_matrix, query_vec.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._threshold:
            key = self._keys[best_idx]
            entry = self._cache[key]
            self._cache.move_to_end(key)
            logger.info("semantic_cache_hit", query=query[:50], score=round(best_score, 4), cached_query=entry.query[:50])
            return entry

        logger.debug("semantic_cache_miss", query=query[:50], best_score=round(best_score, 4))
        return None

    async def put(self, query: str, answer: str, sources: list[dict] | None = None) -> None:
        """写入缓存"""
        query_emb = await self._embedding.embed_query(query)

        if len(self._cache) >= self._max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            idx = self._keys.index(oldest_key)
            self._keys.pop(idx)
            self._embeddings.pop(idx)

        key = f"cache_{len(self._cache)}_{int(time.time())}"
        entry = CacheEntry(
            query=query,
            answer=answer,
            sources=sources or [],
            embedding=query_emb,
            created_at=time.time(),
        )

        self._cache[key] = entry
        self._keys.append(key)
        self._embeddings.append(query_emb)

        logger.info("semantic_cache_stored", query=query[:50], cache_size=len(self._cache))

    def _evict_expired(self) -> None:
        """清除过期条目"""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v.created_at > self._ttl]
        for key in expired_keys:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            self._embeddings.pop(idx)
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
        self._keys.clear()
        self._embeddings.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
