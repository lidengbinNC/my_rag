"""
Reranker 抽象基类

面试考点（高频必考）：
- 两阶段检索架构：Retrieval（召回） → Reranking（精排）
  - 第一阶段（召回）：Bi-Encoder，query 和 doc 独立编码，速度快，从百万级文档中召回 top-K
  - 第二阶段（精排）：Cross-Encoder，query 和 doc 联合编码，精度高但慢，对 top-K 重排序

- Bi-Encoder vs Cross-Encoder 核心区别：
  ┌─────────────┬───────────────────────────┬────────────────────────────┐
  │             │ Bi-Encoder (召回)          │ Cross-Encoder (精排)        │
  ├─────────────┼───────────────────────────┼────────────────────────────┤
  │ 输入方式     │ query 和 doc 分别编码       │ [query, doc] 拼接后联合编码  │
  │ 交互层次     │ 无交互（独立向量）          │ 深度交互（attention 互看）    │
  │ 速度        │ 快（向量预计算 + ANN）       │ 慢（每对都要推理）           │
  │ 精度        │ 较低                       │ 高                          │
  │ 适用规模     │ 百万~亿级                   │ top-K（通常 20~100）         │
  └─────────────┴───────────────────────────┴────────────────────────────┘

- 为什么 Reranker 能提升效果？
  Bi-Encoder 的向量是独立编码的，无法捕捉 query-doc 之间的细粒度交互。
  Cross-Encoder 让 query 和 doc 的 token 在 Transformer 的每一层都能互相 attend，
  因此能理解更复杂的语义关系（如否定、条件、对比）。

- 常用 Reranker 模型：
  - BAAI/bge-reranker-v2-m3（开源，支持中英文）
  - Cohere Rerank API（商用 API）
  - LLM-based Reranker（用 GPT/Qwen 打分，灵活但慢）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from my_rag.domain.retrieval.base import RetrievalResult


@dataclass
class RerankResult:
    """重排序结果"""
    chunk_id: str
    content: str
    score: float
    original_rank: int
    new_rank: int
    source: str = ""
    metadata: dict | None = None


class BaseReranker(ABC):

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        对检索结果重排序

        Args:
            query: 用户查询
            results: 初始检索结果（已按第一阶段分数排序）
            top_k: 重排序后保留的数量（None 表示保留全部）

        Returns:
            按新分数降序排列的重排序结果
        """
        ...
