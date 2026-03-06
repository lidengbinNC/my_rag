"""Tests for the main RAG Pipeline (run + stream)."""

import pytest

from my_rag.core.rag_pipeline import RAGPipeline, RAGResponse
from my_rag.domain.retrieval.base import RetrievalResult
from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from my_rag.core.semantic_cache import SemanticCache
from tests.conftest import (
    FakeEmbedding,
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    make_retrieval_results,
)


@pytest.fixture
def pipeline(fake_retriever, fake_llm) -> RAGPipeline:
    return RAGPipeline(retriever=fake_retriever, llm=fake_llm)


class TestRAGPipelineRun:
    @pytest.mark.asyncio
    async def test_basic_run(self, pipeline: RAGPipeline):
        """基础 RAG 流程：检索 → 生成。"""
        resp = await pipeline.run("什么是 RAG？", knowledge_base_id="kb_1")
        assert isinstance(resp, RAGResponse)
        assert resp.answer == "fake answer"
        assert len(resp.sources) > 0
        assert resp.cached is False

    @pytest.mark.asyncio
    async def test_prompt_contains_context(self, pipeline: RAGPipeline, fake_llm: FakeLLM):
        """生成时的 prompt 应包含检索到的文档内容。"""
        await pipeline.run("test", knowledge_base_id="kb_1")
        prompt_sent = fake_llm.call_history[-1]
        assert "测试文档" in prompt_sent

    @pytest.mark.asyncio
    async def test_empty_retrieval(self, fake_llm):
        """检索结果为空时仍应正常生成。"""
        empty_retriever = FakeRetriever([])
        p = RAGPipeline(retriever=empty_retriever, llm=fake_llm)
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert resp.answer == "fake answer"
        assert resp.sources == []


class TestRAGPipelineWithCache:
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, fake_retriever, fake_llm):
        """第一次查询 miss，第二次相同查询 hit。"""
        cache = SemanticCache(FakeEmbedding(dim=4), similarity_threshold=0.90)
        p = RAGPipeline(
            retriever=fake_retriever, llm=fake_llm,
            semantic_cache=cache, enable_cache=True,
        )

        resp1 = await p.run("什么是 RAG", knowledge_base_id="kb_1")
        assert resp1.cached is False

        resp2 = await p.run("什么是 RAG", knowledge_base_id="kb_1")
        assert resp2.cached is True
        assert resp2.answer == resp1.answer

    @pytest.mark.asyncio
    async def test_cache_disabled(self, fake_retriever, fake_llm):
        """enable_cache=False 时不走缓存。"""
        cache = SemanticCache(FakeEmbedding(dim=4))
        p = RAGPipeline(
            retriever=fake_retriever, llm=fake_llm,
            semantic_cache=cache, enable_cache=False,
        )
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert resp.cached is False
        assert cache.size == 0


class TestRAGPipelineWithRerank:
    @pytest.mark.asyncio
    async def test_reranker_applied(self, fake_retriever, fake_llm):
        """启用 reranker 时应改变文档顺序。"""
        reranker = FakeReranker()
        p = RAGPipeline(
            retriever=fake_retriever, llm=fake_llm,
            reranker=reranker, rerank_top_k=3,
        )
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert len(resp.sources) == 3  # rerank_top_k=3


class TestRAGPipelineWithRewrite:
    @pytest.mark.asyncio
    async def test_hyde_rewrite(self, fake_retriever):
        """HyDE 模式：LLM 先生成假设性文档再检索。"""
        llm = FakeLLM(responses=["假设性文档内容", "最终答案"])
        rewriter = QueryRewriter(llm)
        p = RAGPipeline(
            retriever=fake_retriever, llm=llm,
            query_rewriter=rewriter, enable_hyde=True,
        )
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert resp.answer == "最终答案"

    @pytest.mark.asyncio
    async def test_multi_query_rewrite(self, fake_retriever):
        """Multi-Query 模式：生成多个子查询分别检索。"""
        llm = FakeLLM(responses=["子查询1\n子查询2\n子查询3", "最终答案"])
        rewriter = QueryRewriter(llm)
        p = RAGPipeline(
            retriever=fake_retriever, llm=llm,
            query_rewriter=rewriter, enable_multi_query=True,
        )
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert resp.answer == "最终答案"
        assert fake_retriever.call_count >= 2  # original + sub-queries


class TestRAGPipelineSelfRAG:
    @pytest.mark.asyncio
    async def test_skip_retrieval_when_not_needed(self):
        """Self-RAG: LLM 判断不需要检索时跳过检索。"""
        llm = FakeLLM(responses=["否", "直接回答"])
        retriever = FakeRetriever(make_retrieval_results(3))
        p = RAGPipeline(
            retriever=retriever, llm=llm, enable_self_rag=True,
        )
        resp = await p.run("你好", knowledge_base_id="kb_1")
        assert resp.needed_retrieval is False
        assert retriever.call_count == 0

    @pytest.mark.asyncio
    async def test_proceeds_when_retrieval_needed(self):
        """Self-RAG: 判断需要检索 → 检索 → 相关性过滤 → 生成 → 支撑性检查。"""
        llm = FakeLLM(responses=[
            "是",          # need retrieval → yes
            "相关",        # doc 0 relevant
            "相关",        # doc 1 relevant
            "不相关",      # doc 2 not relevant
            "不相关",      # doc 3 not relevant
            "不相关",      # doc 4 not relevant
            "fake answer", # generation
            "支撑",        # support check → supported
        ])
        retriever = FakeRetriever(make_retrieval_results(5))
        p = RAGPipeline(
            retriever=retriever, llm=llm, enable_self_rag=True,
        )
        resp = await p.run("什么是 RAG？", knowledge_base_id="kb_1")
        assert resp.needed_retrieval is True
        assert resp.is_supported is True
        assert resp.relevant_sources is not None
        assert len(resp.relevant_sources) == 2

    @pytest.mark.asyncio
    async def test_retry_on_unsupported_answer(self):
        """Self-RAG: 不支撑时重试生成。"""
        llm = FakeLLM(responses=[
            "是",          # need retrieval
            "相关", "相关", "相关", "相关", "相关",  # all relevant
            "first answer",    # generation attempt 1
            "不支撑",          # not supported → retry
            "second answer",   # generation attempt 2
            "支撑",            # supported
        ])
        retriever = FakeRetriever(make_retrieval_results(5))
        p = RAGPipeline(
            retriever=retriever, llm=llm,
            enable_self_rag=True, self_rag_max_retries=1,
        )
        resp = await p.run("query", knowledge_base_id="kb_1")
        assert resp.retried is True
        assert resp.is_supported is True


class TestRAGPipelineParentChild:
    @pytest.mark.asyncio
    async def test_parent_content_used_in_context(self):
        """Parent-Child: 命中 child 时用 parent_content 构建上下文。"""
        sources = [
            RetrievalResult(
                chunk_id="child_0", content="child text",
                score=0.9, source="doc.txt",
                metadata={"parent_content": "full parent context with more info"},
            )
        ]
        llm = FakeLLM()
        retriever = FakeRetriever(sources)
        p = RAGPipeline(retriever=retriever, llm=llm)
        await p.run("query", knowledge_base_id="kb_1")

        prompt = llm.call_history[-1]
        assert "full parent context" in prompt
        assert "child text" not in prompt


class TestRAGPipelineStream:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens_and_done(self, fake_retriever, fake_llm):
        """流式输出应先 yield retrieval、多个 token、最后 done。"""
        p = RAGPipeline(retriever=fake_retriever, llm=fake_llm)
        events = []
        async for event in p.stream("query", knowledge_base_id="kb_1"):
            events.append(event)

        types = [e["type"] for e in events]
        assert types[0] == "retrieval"
        assert "token" in types
        assert types[-1] == "done"
        assert events[-1]["full_answer"] == "fake answer"
