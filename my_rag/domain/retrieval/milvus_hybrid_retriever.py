"""
Milvus 原生 Dense + Sparse 混合检索器。

与“DenseRetriever + BM25 + RRF”不同，这里直接依赖 Milvus 的 hybrid_search，
由向量库统一执行 dense/sparse 两路召回与融合。
"""

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.infrastructure.vector_store.base import BaseVectorStore


class MilvusHybridRetriever(BaseRetriever):

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        ranker: str = "weighted",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ):
        self._embedding = embedding
        self._vector_store = vector_store
        self._ranker = ranker
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        dense_query, sparse_query = await self._embedding.embed_query_hybrid(query)
        filter_meta = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else None
        results = await self._vector_store.search_hybrid(
            query_embedding=dense_query,
            query_sparse_embedding=sparse_query,
            top_k=top_k,
            filter_metadata=filter_meta,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            ranker=self._ranker,
            candidate_limit=self._candidate_limit,
            rrf_k=self._rrf_k,
        )
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                source=r.metadata.get("source", ""),
                metadata=r.metadata,
            )
            for r in results
        ]
