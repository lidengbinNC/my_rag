"""Tests for Corrective RAG pipeline."""

import pytest

from my_rag.core.corrective_rag import (
    CorrectiveRAGPipeline,
    CorrectiveRAGResult,
    RelevanceGrade,
)
from tests.conftest import FakeLLM, FakeRetriever, make_retrieval_results


class TestCorrectiveRAG:
    @pytest.mark.asyncio
    async def test_all_correct_documents(self):
        """所有文档评为 correct 时全部使用。"""
        llm = FakeLLM(responses=[
            "correct", "correct", "correct",  # grading
            "final answer",                    # generation
        ])
        retriever = FakeRetriever(make_retrieval_results(3))
        pipeline = CorrectiveRAGPipeline(retriever, llm)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert isinstance(result, CorrectiveRAGResult)
        assert result.correct_count == 3
        assert result.incorrect_count == 0
        assert result.supplemented is False
        assert result.answer == "final answer"

    @pytest.mark.asyncio
    async def test_all_incorrect_no_supplement(self):
        """所有文档评为 incorrect 且 ambiguous 不足时，返回兜底回答。"""
        llm = FakeLLM(responses=["incorrect", "incorrect", "incorrect"])
        retriever = FakeRetriever(make_retrieval_results(3))
        pipeline = CorrectiveRAGPipeline(retriever, llm)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert result.incorrect_count == 3
        assert "未能找到" in result.answer

    @pytest.mark.asyncio
    async def test_ambiguous_triggers_supplement(self):
        """correct=0 且 ambiguous ≥ threshold 时触发补充检索。"""
        llm = FakeLLM(responses=[
            "ambiguous", "ambiguous", "ambiguous",  # grading: 3 ambiguous
            "supplemented answer",                   # generation
        ])
        retriever = FakeRetriever(make_retrieval_results(3))
        pipeline = CorrectiveRAGPipeline(retriever, llm, ambiguous_threshold=2)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert result.ambiguous_count == 3
        assert result.supplemented is True
        assert retriever.call_count == 2  # initial + supplement

    @pytest.mark.asyncio
    async def test_mixed_grades(self):
        """混合评分时只用 correct + ambiguous 文档。"""
        llm = FakeLLM(responses=[
            "correct",     # doc 0
            "ambiguous",   # doc 1
            "incorrect",   # doc 2
            "answer",      # generation
        ])
        retriever = FakeRetriever(make_retrieval_results(3))
        pipeline = CorrectiveRAGPipeline(retriever, llm)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert result.correct_count == 1
        assert result.ambiguous_count == 1
        assert result.incorrect_count == 1
        assert len(result.sources) == 2  # correct + ambiguous

    @pytest.mark.asyncio
    async def test_grading_exception_defaults_to_ambiguous(self):
        """评分出异常时默认为 ambiguous。"""

        class FailingLLM(FakeLLM):
            _count = 0

            async def generate(self, prompt: str, **kwargs) -> str:
                self._count += 1
                if self._count <= 2:
                    raise RuntimeError("LLM error")
                return await super().generate(prompt, **kwargs)

        llm = FailingLLM(responses=["correct", "answer"])
        retriever = FakeRetriever(make_retrieval_results(3))
        pipeline = CorrectiveRAGPipeline(retriever, llm)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert result.ambiguous_count >= 2

    @pytest.mark.asyncio
    async def test_graded_documents_in_result(self):
        """结果应包含每篇文档的评分详情。"""
        llm = FakeLLM(responses=["correct", "incorrect", "answer"])
        retriever = FakeRetriever(make_retrieval_results(2))
        pipeline = CorrectiveRAGPipeline(retriever, llm)

        result = await pipeline.run("query", knowledge_base_id="kb_1")
        assert len(result.graded) == 2
        grades = {g.grade for g in result.graded}
        assert RelevanceGrade.CORRECT in grades
        assert RelevanceGrade.INCORRECT in grades
