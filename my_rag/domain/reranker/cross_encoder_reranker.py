"""
Cross-Encoder Reranker（本地模型版）

面试考点：
- Cross-Encoder 原理：将 [query, document] 作为一个整体输入 BERT/Transformer，
  输出单个相关性分数。与 Bi-Encoder 的区别是 query 和 doc 的 token 在 attention 层
  可以互相看到，因此能捕捉更精细的语义交互。

- 为什么用 Sigmoid？
  Cross-Encoder 的原始 logit 范围不固定，Sigmoid 将其映射到 [0, 1] 作为相关性概率，
  方便设置统一阈值过滤低质量结果。

- 性能优化：
  - asyncio.to_thread：CPU 密集的模型推理放入线程池，不阻塞 event loop
  - batch predict：一次性预测所有 (query, doc) 对，利用 GPU/CPU 批处理加速
  - score_threshold：过滤低于阈值的结果，减少传给 LLM 的噪声上下文
"""

import asyncio
from functools import lru_cache

from my_rag.domain.reranker.base import BaseReranker, RerankResult
from my_rag.domain.retrieval.base import RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    """延迟加载并缓存 Cross-Encoder 模型（面试点：lru_cache 实现模型单例）"""
    from sentence_transformers import CrossEncoder
    logger.info("loading_cross_encoder", model=model_name)
    return CrossEncoder(model_name)


class CrossEncoderReranker(BaseReranker):
    """
    基于 Cross-Encoder 的重排序器

    典型使用流程：
    1. HybridRetriever 召回 top-20 候选文档
    2. CrossEncoderReranker 对 20 个 (query, doc) 对联合编码
    3. 按 Cross-Encoder 分数重排，取 top-5 送给 LLM
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        score_threshold: float = 0.0,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._score_threshold = score_threshold
        self._max_length = max_length

    def _get_model(self):
        return _load_cross_encoder(self._model_name)

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not results:
            return []

        model = self._get_model()

        pairs = [[query, r.content[:self._max_length]] for r in results]

        scores = await asyncio.to_thread(model.predict, pairs)

        scored_results: list[RerankResult] = []
        for i, (result, score) in enumerate(zip(results, scores)):
            raw_score = float(score)
            norm_score = 1.0 / (1.0 + _neg_exp(raw_score))

            if norm_score < self._score_threshold:
                continue

            scored_results.append(RerankResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=norm_score,
                original_rank=i,
                new_rank=0,
                source=result.source,
                metadata=result.metadata,
            ))

        scored_results.sort(key=lambda r: r.score, reverse=True)

        for rank, r in enumerate(scored_results):
            r.new_rank = rank

        if top_k is not None:
            scored_results = scored_results[:top_k]

        logger.info(
            "cross_encoder_rerank_done",
            input_count=len(results),
            output_count=len(scored_results),
            top_score=round(scored_results[0].score, 4) if scored_results else 0,
        )

        return scored_results


def _neg_exp(x: float) -> float:
    """安全的 exp(-x)，防止溢出"""
    import math
    try:
        return math.exp(-x)
    except OverflowError:
        return 0.0 if x > 0 else float("inf")
