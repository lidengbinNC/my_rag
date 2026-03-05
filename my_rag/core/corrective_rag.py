"""
Corrective RAG（纠正式 RAG）

面试考点（区分度极高）：
- 传统 RAG 的问题：检索到的文档不一定都相关，直接全部丢给 LLM 会引入噪声
- Corrective RAG 核心思想：对每篇检索文档做相关性打分，分三档处理
  1. Correct（相关）：直接使用
  2. Ambiguous（模糊）：用更多文档补充（扩大检索范围）
  3. Incorrect（不相关）：丢弃，触发补充策略

- 论文：Corrective Retrieval Augmented Generation (Yan et al., 2024)
- 与 Self-RAG 的区别：
  - Corrective RAG 聚焦检索端的质量控制（检索后、生成前）
  - Self-RAG 聚焦生成端的自我反思（生成后评估）
- 两者可组合使用：先 Corrective 清洗文档 → 再 Self-RAG 检查回答
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.prompt.template import build_context, build_prompt
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class RelevanceGrade(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


GRADE_PROMPT = """你是一个文档相关性评估专家。请评估以下文档片段与用户问题的相关程度。

用户问题：{query}

文档片段：
{document}

请严格按以下标准评分：
- "correct"：文档直接包含回答问题所需的信息
- "ambiguous"：文档部分相关，但信息不够完整
- "incorrect"：文档与问题完全不相关

请只输出一个词：correct、ambiguous 或 incorrect"""


@dataclass
class GradedDocument:
    result: RetrievalResult
    grade: RelevanceGrade


@dataclass
class CorrectiveRAGResult:
    answer: str
    sources: list[RetrievalResult]
    graded: list[GradedDocument] = field(default_factory=list)
    correct_count: int = 0
    ambiguous_count: int = 0
    incorrect_count: int = 0
    supplemented: bool = False


class CorrectiveRAGPipeline:
    """Corrective RAG：检索 → 文档评分 → 过滤/补充 → 生成"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        ambiguous_threshold: int = 2,
    ):
        self._retriever = retriever
        self._llm = llm
        self._ambiguous_threshold = ambiguous_threshold

    async def run(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> CorrectiveRAGResult:

        sources = await self._retriever.retrieve(query, top_k=top_k, knowledge_base_id=knowledge_base_id)

        graded_docs = await self._grade_documents(query, sources)

        correct = [g for g in graded_docs if g.grade == RelevanceGrade.CORRECT]
        ambiguous = [g for g in graded_docs if g.grade == RelevanceGrade.AMBIGUOUS]
        incorrect = [g for g in graded_docs if g.grade == RelevanceGrade.INCORRECT]

        logger.info(
            "corrective_rag_grading",
            correct=len(correct),
            ambiguous=len(ambiguous),
            incorrect=len(incorrect),
        )

        supplemented = False
        final_sources = [g.result for g in correct]

        if len(correct) == 0 and len(ambiguous) >= self._ambiguous_threshold:
            supplement = await self._retriever.retrieve(query, top_k=top_k * 2, knowledge_base_id=knowledge_base_id)
            existing_ids = {s.chunk_id for s in sources}
            new_sources = [s for s in supplement if s.chunk_id not in existing_ids]
            final_sources.extend([g.result for g in ambiguous])
            final_sources.extend(new_sources[:top_k])
            supplemented = True
            logger.info("corrective_rag_supplemented", new_count=len(new_sources))
        elif ambiguous:
            final_sources.extend([g.result for g in ambiguous])

        if not final_sources:
            return CorrectiveRAGResult(
                answer="根据已有文档，未能找到足够相关的内容来回答您的问题。建议补充更多相关文档后再尝试。",
                sources=sources, graded=graded_docs,
                correct_count=len(correct), ambiguous_count=len(ambiguous), incorrect_count=len(incorrect),
            )

        context = build_context([{"content": s.content, "source": s.source} for s in final_sources])
        prompt = build_prompt(question=query, context=context, chat_history=chat_history)
        answer = await self._llm.generate(prompt)

        return CorrectiveRAGResult(
            answer=answer, sources=final_sources, graded=graded_docs,
            correct_count=len(correct), ambiguous_count=len(ambiguous),
            incorrect_count=len(incorrect), supplemented=supplemented,
        )

    async def _grade_documents(self, query: str, sources: list[RetrievalResult]) -> list[GradedDocument]:
        """并发为每篇文档打分"""
        tasks = [self._grade_single(query, source) for source in sources]
        return await asyncio.gather(*tasks)

    async def _grade_single(self, query: str, source: RetrievalResult) -> GradedDocument:
        """单篇文档评分"""
        try:
            result = await self._llm.generate(
                GRADE_PROMPT.format(query=query, document=source.content[:500]),
                temperature=0.0, max_tokens=10,
            )
            text = result.strip().lower()
            if "correct" in text and "incorrect" not in text:
                grade = RelevanceGrade.CORRECT
            elif "incorrect" in text:
                grade = RelevanceGrade.INCORRECT
            else:
                grade = RelevanceGrade.AMBIGUOUS
        except Exception:
            grade = RelevanceGrade.AMBIGUOUS

        return GradedDocument(result=source, grade=grade)
