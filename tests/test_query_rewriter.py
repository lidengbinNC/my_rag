"""Tests for QueryRewriter (HyDE / Multi-Query)."""

import pytest

from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from tests.conftest import FakeLLM


class TestHyDE:
    @pytest.mark.asyncio
    async def test_returns_hypothetical_document(self):
        llm = FakeLLM(responses=["这是一段关于 RAG 的假设性文档内容。"])
        rewriter = QueryRewriter(llm)
        result = await rewriter.hyde("什么是 RAG？")
        assert "假设性文档" in result

    @pytest.mark.asyncio
    async def test_fallback_to_original_on_error(self):
        """LLM 异常时应返回原始查询。"""

        class ErrorLLM(FakeLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise RuntimeError("API error")

        rewriter = QueryRewriter(ErrorLLM())
        result = await rewriter.hyde("original query")
        assert result == "original query"


class TestMultiQuery:
    @pytest.mark.asyncio
    async def test_generates_sub_queries(self):
        llm = FakeLLM(responses=["子问题1\n子问题2\n子问题3"])
        rewriter = QueryRewriter(llm)
        queries = await rewriter.multi_query("什么是 RAG？")

        assert queries[0] == "什么是 RAG？"  # original always first
        assert len(queries) == 4  # original + 3 sub-queries

    @pytest.mark.asyncio
    async def test_caps_at_3_sub_queries(self):
        """即使 LLM 返回更多行，也只取前 3 个子查询。"""
        llm = FakeLLM(responses=["q1\nq2\nq3\nq4\nq5"])
        rewriter = QueryRewriter(llm)
        queries = await rewriter.multi_query("test")
        assert len(queries) == 4  # 1 original + 3 max

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """LLM 异常时应只返回原始查询。"""

        class ErrorLLM(FakeLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise RuntimeError("fail")

        rewriter = QueryRewriter(ErrorLLM())
        queries = await rewriter.multi_query("original")
        assert queries == ["original"]

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """LLM 返回空白时应只返回原始查询。"""
        llm = FakeLLM(responses=[""])
        rewriter = QueryRewriter(llm)
        queries = await rewriter.multi_query("test")
        assert queries == ["test"]
