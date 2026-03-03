"""
查询改写模块

面试考点（高频）：
- 用户原始 query 往往简短、含歧义、有指代，直接检索效果差
- 查询改写是 Advanced RAG 的核心优化手段

1. HyDE (Hypothetical Document Embedding)
   - 原理：让 LLM 先生成一段"假设性回答"，再用这段回答去做向量检索
   - 为什么有效？假设性回答比短 query 包含更多语义信息，与真实文档的向量距离更近
   - 论文：Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)

2. Multi-Query
   - 原理：将原始 query 改写为多个不同角度的子查询，分别检索后合并去重
   - 为什么有效？覆盖查询的多个语义维度，提升 recall
   - 举例："RAG 怎么优化？" → ["RAG 检索阶段如何提升精度？", "RAG 生成阶段如何减少幻觉？", "RAG 有哪些评估指标？"]
"""

from my_rag.domain.llm.base import BaseLLM
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

HYDE_PROMPT = """请根据以下问题，写一段可能包含答案的文档段落（约 100-200 字）。
不需要真的回答问题，只需要生成一段看起来像是来自相关文档的文字。

问题：{query}

文档段落："""

MULTI_QUERY_PROMPT = """你是一个查询优化助手。请将用户的原始问题改写为 3 个不同角度的子问题，
用于从文档库中检索更全面的信息。

规则：
1. 每个子问题从不同维度理解原始问题
2. 子问题应该更具体、更利于检索
3. 每行一个子问题，不要编号

原始问题：{query}

改写后的子问题："""


class QueryRewriter:
    """查询改写器"""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    async def hyde(self, query: str) -> str:
        """
        HyDE: 生成假设性文档，用于替代原始 query 做检索
        返回假设性文档文本
        """
        try:
            prompt = HYDE_PROMPT.format(query=query)
            hypothetical_doc = await self._llm.generate(prompt, max_tokens=300, temperature=0.7)
            logger.info("hyde_generated", query=query[:50], doc_length=len(hypothetical_doc))
            return hypothetical_doc.strip()
        except Exception as e:
            logger.warning("hyde_failed", error=str(e))
            return query

    async def multi_query(self, query: str) -> list[str]:
        """
        Multi-Query: 将原始问题改写为多个子问题
        返回子问题列表（包含原始 query）
        """
        try:
            prompt = MULTI_QUERY_PROMPT.format(query=query)
            result = await self._llm.generate(prompt, max_tokens=300, temperature=0.7)
            sub_queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
            sub_queries = [query] + sub_queries[:3]
            logger.info("multi_query_generated", original=query[:50], sub_query_count=len(sub_queries))
            return sub_queries
        except Exception as e:
            logger.warning("multi_query_failed", error=str(e))
            return [query]
