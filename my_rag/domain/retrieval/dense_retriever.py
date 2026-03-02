"""
稠密向量检索器

面试考点：
- 稠密检索将 query 和 doc 都映射到同一向量空间，用向量距离衡量相关性
- 优势：能理解语义，"机器学习" 可以匹配 "ML"、"深度学习" 等
- 劣势：对精确关键词匹配（如产品型号 ID）效果差
"""

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.infrastructure.vector_store.base import BaseVectorStore


class DenseRetriever(BaseRetriever):

    def __init__(self, embedding: BaseEmbedding, vector_store: BaseVectorStore):
        self._embedding = embedding
        self._vector_store = vector_store

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        query_embedding = await self._embedding.embed_query(query)

        filter_meta = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else None
        results = await self._vector_store.search(query_embedding, top_k=top_k, filter_metadata=filter_meta)

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
