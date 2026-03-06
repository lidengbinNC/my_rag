"""
LLM-based Reranker（基于大模型打分的重排序器）

面试考点：
- LLM Reranker vs Cross-Encoder Reranker：
  ┌───────────────┬──────────────────────┬──────────────────────┐
  │               │ Cross-Encoder        │ LLM Reranker          │
  ├───────────────┼──────────────────────┼──────────────────────┤
  │ 模型大小       │ 小（~300MB）          │ 大（API 调用）         │
  │ 推理速度       │ 快（本地 batch）      │ 慢（逐条 API 调用）   │
  │ 精度          │ 高                    │ 更高（理解能力更强）   │
  │ 成本          │ 低（本地推理）         │ 高（API token 费用）  │
  │ 灵活性        │ 需要针对性微调         │ 零样本，prompt 可调   │
  │ 适用场景       │ 大批量、低延迟需求     │ 小批量、高精度需求     │
  └───────────────┴──────────────────────┴──────────────────────┘

- 并发控制：asyncio.Semaphore 限制同时发给 LLM 的请求数，避免 rate limit
- Prompt 设计：要求 LLM 返回结构化的 JSON 分数，方便解析
"""

import asyncio
import json
import re

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.reranker.base import BaseReranker, RerankResult
from my_rag.domain.retrieval.base import RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

RERANK_SCORE_PROMPT = """你是一个文档相关性评分专家。请判断以下文档与用户问题的相关程度。

## 评分标准
- 10 分：文档直接且完整地回答了问题
- 7-9 分：文档高度相关，包含回答问题的关键信息
- 4-6 分：文档部分相关，包含一些有用信息
- 1-3 分：文档与问题弱相关
- 0 分：文档与问题完全不相关

## 用户问题
{query}

## 文档内容
{document}

请只返回一个 JSON 对象，格式如下（不要返回其他内容）：
{{"score": <0-10的整数>, "reason": "<一句话理由>"}}"""


class LLMReranker(BaseReranker):
    """
    基于 LLM 的重排序器

    适用场景：
    - 没有本地 GPU / 不想部署 Cross-Encoder 模型
    - 文档数量较少（<20 篇），可以承受多次 LLM 调用
    - 需要可解释的打分理由（reason 字段）
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_concurrency: int = 5,
        score_threshold: float = 3.0,
        max_doc_length: int = 800,
    ):
        self._llm = llm
        self._max_concurrency = max_concurrency
        self._score_threshold = score_threshold
        self._max_doc_length = max_doc_length

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not results:
            return []

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def score_one(index: int, result: RetrievalResult) -> RerankResult:
            async with semaphore:
                score = await self._get_score(query, result.content)
                return RerankResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=score / 10.0,
                    original_rank=index,
                    new_rank=0,
                    source=result.source,
                    metadata=result.metadata,
                )

        tasks = [score_one(i, r) for i, r in enumerate(results)]
        scored_results = await asyncio.gather(*tasks)

        scored_results = [r for r in scored_results if r.score >= self._score_threshold / 10.0]

        scored_results.sort(key=lambda r: r.score, reverse=True)

        for rank, r in enumerate(scored_results):
            r.new_rank = rank

        if top_k is not None:
            scored_results = scored_results[:top_k]

        logger.info(
            "llm_rerank_done",
            input_count=len(results),
            output_count=len(scored_results),
            top_score=round(scored_results[0].score, 4) if scored_results else 0,
        )

        return scored_results

    async def _get_score(self, query: str, document: str) -> float:
        """调用 LLM 获取单个文档的相关性分数"""
        prompt = RERANK_SCORE_PROMPT.format(
            query=query,
            document=document[:self._max_doc_length],
        )
        try:
            response = await self._llm.generate(prompt, temperature=0.0, max_tokens=100)
            return self._parse_score(response)
        except Exception as e:
            logger.warning("llm_rerank_score_failed", error=str(e))
            return 5.0

    @staticmethod
    def _parse_score(response: str) -> float:
        """从 LLM 响应中提取分数"""
        try:
            data = json.loads(response)
            return float(data.get("score", 5))
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response)
        if match:
            return float(match.group(1))

        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        for n in numbers:
            val = float(n)
            if 0 <= val <= 10:
                return val

        return 5.0
