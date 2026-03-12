"""
RAG Pipeline 编排器（Phase 4 升级版 + Self-RAG）

面试考点：
- Pipeline 模式：Query → (Cache Check) → (Rewrite) → Retrieve → (Rerank 精排) → Generate → (Cache Store)
- 各环节可独立开关：通过配置决定是否启用 HyDE / Multi-Query / Cache / Self-RAG
- Parent-Child 策略：检索命中 child chunk 时，返回 parent_content 给 LLM
- 流式输出：AsyncIterator 逐 token 传递给前端
- Self-RAG 三层 LLM 判断（可配置 enable_self_rag）：
  1. 检索判断（Retrieve Token）：这个问题需要检索吗？
  2. 相关性判断（ISREL Token）：检索到的文档与问题是否相关？过滤噪声
  3. 支撑性判断（ISSUP Token）：生成的回答是否被文档支撑？检测幻觉
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.prompt.template import build_context, build_prompt
from my_rag.domain.reranker.base import BaseReranker
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from my_rag.core.semantic_cache import SemanticCache
from my_rag.utils.logger import get_logger
from my_rag.utils.tracing import Trace
from my_rag.utils import metrics as prom

logger = get_logger(__name__)

# ── Self-RAG prompt 模板 ──────────────────────────────────────────────

NEED_RETRIEVAL_PROMPT = """判断以下问题是否需要从文档库中检索信息才能回答。

如果是常识性问题、闲聊、或不需要特定文档支持的问题，回答"否"。
如果需要查阅特定文档、数据或专业知识才能准确回答，回答"是"。

问题：{query}

请只回答"是"或"否"，不要解释："""

RELEVANCE_JUDGE_PROMPT = """判断以下文档片段是否与用户问题相关。

用户问题：{query}

文档片段：
{document}

请只回答"相关"或"不相关"："""

SUPPORT_JUDGE_PROMPT = """判断以下回答是否完全由参考文档支撑，没有编造信息。

参考文档：
{context}

回答：
{answer}

请只回答"支撑"或"不支撑"，不要解释："""


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]
    prompt: str = ""
    usage: dict = field(default_factory=dict)
    cached: bool = False
    # Self-RAG 扩展字段（enable_self_rag=True 时才有意义）
    needed_retrieval: bool | None = None
    relevant_sources: list[RetrievalResult] | None = None
    is_supported: bool | None = None
    retried: bool = False


class RAGPipeline:
    """RAG Pipeline：(Self-RAG 判断) → (Cache) → (Rewrite) → Retrieve → (Rerank) → (相关性过滤) → Prompt → Generate → (支撑性检查) → (Cache Store)"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        query_rewriter: QueryRewriter | None = None,
        semantic_cache: SemanticCache | None = None,
        reranker: BaseReranker | None = None,
        enable_hyde: bool = False,
        enable_multi_query: bool = False,
        enable_cache: bool = True,
        enable_self_rag: bool = False,
        self_rag_max_retries: int = 1,
        rerank_top_k: int = 5,
    ):
        self._retriever = retriever
        self._llm = llm
        self._rewriter = query_rewriter
        self._cache = semantic_cache
        self._reranker = reranker
        self._enable_hyde = enable_hyde
        self._enable_multi_query = enable_multi_query
        self._enable_cache = enable_cache and semantic_cache is not None
        self._enable_self_rag = enable_self_rag
        self._self_rag_max_retries = self_rag_max_retries
        self._rerank_top_k = rerank_top_k
        self._retriever_type = type(retriever).__name__.lower().replace("retriever", "")
        self._reranker_type = type(reranker).__name__.lower().replace("reranker", "") if reranker else ""

    async def run(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> RAGResponse:
        """同步执行完整 RAG 流程（带链路追踪 + Prometheus 指标 + 可选 Self-RAG）"""
        trace = Trace()

        with trace.span("rag_pipeline", query=query[:50]):

            # ── Self-RAG 第 1 层：检索必要性判断 ──
            if self._enable_self_rag:
                with trace.span("self_rag_need_retrieval"):
                    needed = await self._judge_need_retrieval(query)
                if not needed:
                    logger.info("self_rag_skip_retrieval", query=query[:50])
                    answer = await self._llm.generate(
                        f"请回答以下问题：\n{query}" if not chat_history else
                        f"对话历史：\n{chat_history}\n\n请回答：\n{query}"
                    )
                    return RAGResponse(
                        answer=answer, sources=[], needed_retrieval=False,
                        is_supported=True,
                    )

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

            # ── Rerank：精排 ──
            if self._reranker and sources:
                with trace.span("rerank"):
                    _rerank_start = time.perf_counter()
                    rerank_results = await self._reranker.rerank(
                        query, sources, top_k=self._rerank_top_k,
                    )
                    _rerank_dur = time.perf_counter() - _rerank_start
                    prom.RERANK_DURATION.labels(reranker_type=self._reranker_type).observe(_rerank_dur)
                    prom.RERANK_INPUT_DOCS.observe(len(sources))
                    prom.RERANK_OUTPUT_DOCS.observe(len(rerank_results))
                    sources = [
                        RetrievalResult(
                            chunk_id=rr.chunk_id, content=rr.content,
                            score=rr.score, source=rr.source, metadata=rr.metadata or {},
                        )
                        for rr in rerank_results
                    ]
                logger.info(
                    "rag_rerank_done",
                    input_count=len(sources) + len(rerank_results) - len(sources),
                    output_count=len(sources),
                )

            # ── Self-RAG 第 2 层：相关性过滤 ──
            relevant_sources = None
            if self._enable_self_rag:
                with trace.span("self_rag_relevance_filter"):
                    relevant_sources = await self._filter_relevant(query, sources)
                logger.info(
                    "self_rag_relevance_filter",
                    total=len(sources), relevant=len(relevant_sources),
                )
                build_sources = relevant_sources if relevant_sources else sources
            else:
                build_sources = sources

            with trace.span("prompt_build"):
                context = self._build_context_from_sources(build_sources)
                prompt = build_prompt(question=query, context=context, chat_history=chat_history)

            # ── Self-RAG 第 3 层：生成 + 支撑性检查（可重试） ──
            is_supported = None
            retried = False
            max_attempts = (1 + self._self_rag_max_retries) if self._enable_self_rag else 1

            for attempt in range(max_attempts):
                with trace.span("llm_generate", attempt=attempt):
                    _llm_start = time.perf_counter()
                    answer = await self._llm.generate(prompt)
                    prom.LLM_DURATION.observe(time.perf_counter() - _llm_start)

                if self._enable_self_rag:
                    with trace.span("self_rag_support_check", attempt=attempt):
                        is_supported = await self._judge_support(context, answer)
                    logger.info("self_rag_support_check", attempt=attempt, is_supported=is_supported)
                    if is_supported or attempt >= self._self_rag_max_retries:
                        retried = attempt > 0
                        break
                    logger.warning("self_rag_retry", attempt=attempt, reason="answer_not_supported")
                else:
                    break

            logger.info("rag_generation_done", answer_length=len(answer))

            with trace.span("cache_store"):
                if self._enable_cache and self._cache:
                    source_dicts = [{"content": s.content, "source": s.source, "score": s.score} for s in sources]
                    await self._cache.put(query, answer, source_dicts)
                    prom.CACHE_SIZE.set(self._cache.size)

        trace.log_summary()

        return RAGResponse(
            answer=answer, sources=sources, prompt=prompt,
            needed_retrieval=True if self._enable_self_rag else None,
            relevant_sources=relevant_sources,
            is_supported=is_supported,
            retried=retried,
        )

    async def stream(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> AsyncIterator[dict]:
        """流式 RAG（含可选 Self-RAG 三层判断）"""

        # ── Self-RAG 第 1 层：检索必要性判断 ──
        if self._enable_self_rag:
            needed = await self._judge_need_retrieval(query)
            if not needed:
                logger.info("self_rag_skip_retrieval_stream", query=query[:50])
                full_answer = ""
                direct_prompt = (
                    f"请回答以下问题：\n{query}" if not chat_history else
                    f"对话历史：\n{chat_history}\n\n请回答：\n{query}"
                )
                async for token in self._llm.stream_generate(direct_prompt):
                    full_answer += token
                    yield {"type": "token", "content": token}
                yield {
                    "type": "done", "full_answer": full_answer,
                    "sources": [], "needed_retrieval": False,
                }
                return

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

        # ── Rerank：精排 ──
        if self._reranker and sources:
            _rerank_start = time.perf_counter()
            rerank_results = await self._reranker.rerank(
                query, sources, top_k=self._rerank_top_k,
            )
            _rerank_dur = time.perf_counter() - _rerank_start
            prom.RERANK_DURATION.labels(reranker_type=self._reranker_type).observe(_rerank_dur)
            prom.RERANK_INPUT_DOCS.observe(len(sources))
            prom.RERANK_OUTPUT_DOCS.observe(len(rerank_results))
            sources = [
                RetrievalResult(
                    chunk_id=rr.chunk_id, content=rr.content,
                    score=rr.score, source=rr.source, metadata=rr.metadata or {},
                )
                for rr in rerank_results
            ]
            logger.info("rag_rerank_done_stream", output_count=len(sources))

        # ── Self-RAG 第 2 层：相关性过滤 ──
        if self._enable_self_rag:
            relevant_sources = await self._filter_relevant(query, sources)
            logger.info("self_rag_relevance_filter_stream", total=len(sources), relevant=len(relevant_sources))
            build_sources = relevant_sources if relevant_sources else sources
        else:
            relevant_sources = None
            build_sources = sources

        yield {
            "type": "retrieval",
            "documents": [
                {"content": s.content, "source": s.source, "score": s.score, "chunk_id": s.chunk_id}
                for s in build_sources
            ],
        }

        context = self._build_context_from_sources(build_sources)
        prompt = build_prompt(question=query, context=context, chat_history=chat_history)
        # ── Self-RAG 第 3 层：生成 + 支撑性检查（可重试） ──

        is_supported = None
        max_attempts = (1 + self._self_rag_max_retries) if self._enable_self_rag else 1

        for attempt in range(max_attempts):
            full_answer = ""
            _llm_start = time.perf_counter()
            if attempt < max_attempts - 1 and self._enable_self_rag:
                full_answer = await self._llm.generate(prompt)
            else:
                async for token in self._llm.stream_generate(prompt):
                    full_answer += token
                    yield {"type": "token", "content": token}
            prom.LLM_DURATION.observe(time.perf_counter() - _llm_start)

            if self._enable_self_rag:
                is_supported = await self._judge_support(context, full_answer)
                logger.info("self_rag_support_check_stream", attempt=attempt, is_supported=is_supported)
                if is_supported or attempt >= self._self_rag_max_retries:
                    if attempt < max_attempts - 1 or is_supported:
                        pass
                    break
                logger.warning("self_rag_retry_stream", attempt=attempt, reason="answer_not_supported")
            else:
                break

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
            **({"is_supported": is_supported} if self._enable_self_rag else {}),
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

    # ── Self-RAG 三层 LLM 判断 ──────────────────────────────────────

    async def _judge_need_retrieval(self, query: str) -> bool:
        """第 1 层（Retrieve Token）：判断是否需要检索"""
        try:
            result = await self._llm.generate(
                NEED_RETRIEVAL_PROMPT.format(query=query),
                temperature=0.0, max_tokens=10,
            )
            return "是" in result
        except Exception:
            return True

    async def _filter_relevant(
        self, query: str, sources: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """第 2 层（ISREL Token）：逐条判断文档相关性，过滤噪声"""
        relevant = []
        for source in sources:
            try:
                result = await self._llm.generate(
                    RELEVANCE_JUDGE_PROMPT.format(query=query, document=source.content[:500]),
                    temperature=0.0, max_tokens=10,
                )
                if "相关" in result and "不相关" not in result:
                    relevant.append(source)
            except Exception:
                relevant.append(source)
        return relevant

    async def _judge_support(self, context: str, answer: str) -> bool:
        """第 3 层（ISSUP Token）：判断回答是否被文档支撑，检测幻觉"""
        try:
            result = await self._llm.generate(
                SUPPORT_JUDGE_PROMPT.format(context=context[:2000], answer=answer),
                temperature=0.0, max_tokens=10,
            )
            return "支撑" in result and "不支撑" not in result
        except Exception:
            return True
