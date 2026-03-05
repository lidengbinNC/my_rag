"""
RAG Pipeline 编排器（Phase 4 升级版）

面试考点：
- Pipeline 模式：Query → (Cache Check) → (Rewrite) → Retrieve → (Rerank) → Generate → (Cache Store)
- 各环节可独立开关：通过配置决定是否启用 HyDE / Multi-Query / Cache
- Parent-Child 策略：检索命中 child chunk 时，返回 parent_content 给 LLM
- 流式输出：AsyncIterator 逐 token 传递给前端
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.prompt.template import build_context, build_prompt
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from my_rag.core.semantic_cache import SemanticCache
from my_rag.utils.logger import get_logger
from my_rag.utils.tracing import Trace
from my_rag.utils import metrics as prom

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]
    prompt: str = ""
    usage: dict = field(default_factory=dict)
    cached: bool = False


class RAGPipeline:
    """RAG Pipeline：(Cache) → (Rewrite) → Retrieve → Prompt → Generate → (Cache Store)"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        query_rewriter: QueryRewriter | None = None,
        semantic_cache: SemanticCache | None = None,
        enable_hyde: bool = False,
        enable_multi_query: bool = False,
        enable_cache: bool = True,
    ):
        self._retriever = retriever
        self._llm = llm
        self._rewriter = query_rewriter
        self._cache = semantic_cache
        self._enable_hyde = enable_hyde
        self._enable_multi_query = enable_multi_query
        self._enable_cache = enable_cache and semantic_cache is not None
        self._retriever_type = type(retriever).__name__.lower().replace("retriever", "")

    async def run(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> RAGResponse:
        """同步执行完整 RAG 流程（带链路追踪 + Prometheus 指标）"""
        trace = Trace()

        with trace.span("rag_pipeline", query=query[:50]):

            with trace.span("cache_check"):
                if self._enable_cache and self._cache:
                    cached = await self._cache.get(query)
                    if cached:
                        prom.CACHE_HIT.inc()
                        return RAGResponse(answer=cached.answer, sources=[], cached=True)
                    prom.CACHE_MISS.inc()

            with trace.span("retrieval"):
                _ret_start = time.perf_counter()
                sources = await self._retrieve_with_rewrite(query, top_k, knowledge_base_id)
                _ret_dur = time.perf_counter() - _ret_start
                prom.RETRIEVAL_COUNT.labels(retriever_type=self._retriever_type).inc()
                prom.RETRIEVAL_DURATION.labels(retriever_type=self._retriever_type).observe(_ret_dur)
                prom.RETRIEVAL_DOCS.observe(len(sources))

            logger.info("rag_retrieval_done", query=query[:50], source_count=len(sources))

            with trace.span("prompt_build"):
                context = self._build_context_from_sources(sources)
                prompt = build_prompt(question=query, context=context, chat_history=chat_history)

            with trace.span("llm_generate"):
                _llm_start = time.perf_counter()
                answer = await self._llm.generate(prompt)
                prom.LLM_DURATION.observe(time.perf_counter() - _llm_start)

            logger.info("rag_generation_done", answer_length=len(answer))

            with trace.span("cache_store"):
                if self._enable_cache and self._cache:
                    source_dicts = [{"content": s.content, "source": s.source, "score": s.score} for s in sources]
                    await self._cache.put(query, answer, source_dicts)
                    prom.CACHE_SIZE.set(self._cache.size)

        trace.log_summary()

        return RAGResponse(answer=answer, sources=sources, prompt=prompt)

    async def stream(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> AsyncIterator[dict]:
        """流式 RAG"""

        if self._enable_cache and self._cache:
            cached = await self._cache.get(query)
            if cached:
                prom.CACHE_HIT.inc()
                yield {"type": "retrieval", "documents": cached.sources}
                yield {"type": "token", "content": cached.answer}
                yield {"type": "done", "full_answer": cached.answer, "sources": cached.sources, "cached": True}
                return
            prom.CACHE_MISS.inc()

        _ret_start = time.perf_counter()
        sources = await self._retrieve_with_rewrite(query, top_k, knowledge_base_id)
        _ret_dur = time.perf_counter() - _ret_start
        prom.RETRIEVAL_COUNT.labels(retriever_type=self._retriever_type).inc()
        prom.RETRIEVAL_DURATION.labels(retriever_type=self._retriever_type).observe(_ret_dur)
        prom.RETRIEVAL_DOCS.observe(len(sources))

        yield {
            "type": "retrieval",
            "documents": [
                {"content": s.content, "source": s.source, "score": s.score, "chunk_id": s.chunk_id}
                for s in sources
            ],
        }

        context = self._build_context_from_sources(sources)
        prompt = build_prompt(question=query, context=context, chat_history=chat_history)

        full_answer = ""
        _llm_start = time.perf_counter()
        async for token in self._llm.stream_generate(prompt):
            full_answer += token
            yield {"type": "token", "content": token}
        prom.LLM_DURATION.observe(time.perf_counter() - _llm_start)

        if self._enable_cache and self._cache:
            source_dicts = [{"content": s.content, "source": s.source, "score": s.score} for s in sources]
            await self._cache.put(query, full_answer, source_dicts)
            prom.CACHE_SIZE.set(self._cache.size)

        yield {
            "type": "done",
            "full_answer": full_answer,
            "sources": [
                {"content": s.content, "source": s.source, "score": s.score, "chunk_id": s.chunk_id}
                for s in sources
            ],
        }

    async def _retrieve_with_rewrite(
        self, query: str, top_k: int, knowledge_base_id: str
    ) -> list[RetrievalResult]:
        """带查询改写的检索"""

        if self._rewriter and self._enable_multi_query:
            sub_queries = await self._rewriter.multi_query(query)
            all_results: dict[str, RetrievalResult] = {}
            for sq in sub_queries:
                results = await self._retriever.retrieve(sq, top_k=top_k, knowledge_base_id=knowledge_base_id)
                for r in results:
                    if r.chunk_id not in all_results or r.score > all_results[r.chunk_id].score:
                        all_results[r.chunk_id] = r
            sorted_results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
            return sorted_results[:top_k]

        if self._rewriter and self._enable_hyde:
            hyde_doc = await self._rewriter.hyde(query)
            return await self._retriever.retrieve(hyde_doc, top_k=top_k, knowledge_base_id=knowledge_base_id)

        return await self._retriever.retrieve(query, top_k=top_k, knowledge_base_id=knowledge_base_id)

    @staticmethod
    def _build_context_from_sources(sources: list[RetrievalResult]) -> str:
        """构建上下文，若使用 Parent-Child 策略则用 parent_content"""
        chunks = []
        for s in sources:
            parent_content = s.metadata.get("parent_content")
            content = parent_content if parent_content else s.content
            chunks.append({"content": content, "source": s.source})
        return build_context(chunks)
