"""
混合检索器 + RRF 融合

面试考点（高频必考）：
- 混合检索 = Dense (语义) + Sparse (关键词)，两者互补
- RRF (Reciprocal Rank Fusion) 融合算法：
    score(d) = Σ 1/(k + rank_i(d))
  - k 是平滑常数（通常 60），防止排名第一的文档分数过大
  - rank_i(d) 是文档 d 在第 i 个检索器中的排名（从 1 开始）
  - 最终按融合分数降序排列
- 为什么 RRF 有效？不依赖各检索器的原始分数（分数量纲不同），只用排名
- 对比其他融合方法：加权求和（需要分数归一化）、CombMNZ 等
"""

from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    RRF 核心算法

    ranked_lists: 多个检索器的有序结果列表
    k: 平滑常数，默认 60（原论文推荐值）
    returns: 融合后的排序结果
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[RetrievalResult] = []
    for chunk_id in sorted_ids:
        result = chunk_map[chunk_id]
        fused.append(RetrievalResult(
            chunk_id=result.chunk_id,
            content=result.content,
            score=rrf_scores[chunk_id],
            source=result.source,
            metadata=result.metadata,
        ))

    return fused


class HybridRetriever(BaseRetriever):
    """混合检索器：组合 Dense + Sparse，通过 RRF 融合"""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        rrf_k: int = 60,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf_k = rrf_k

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        import asyncio

        dense_task = self._dense.retrieve(query, top_k=top_k * 2, knowledge_base_id=knowledge_base_id)
        sparse_task = self._sparse.retrieve(query, top_k=top_k * 2, knowledge_base_id=knowledge_base_id)

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        logger.info(
            "hybrid_retrieval",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self._rrf_k,
        )

        return fused[:top_k]
