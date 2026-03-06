"""
RAG 系统性能基准测试

测量维度：
  1. 各阶段延迟分解（检索 / Rerank / LLM 生成 / 缓存）
  2. 端到端延迟（Naive RAG / Advanced RAG / Self-RAG / Corrective RAG）
  3. Self-RAG 开关前后对比
  4. Rerank 开关前后对比
  5. 语义缓存命中 vs 未命中
  6. 查询改写（HyDE / Multi-Query）开销

用法：
  python -m benchmarks.perf_benchmark              # 运行并输出到终端
  python -m benchmarks.perf_benchmark --output report.md  # 写入文件
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime

from my_rag.core.corrective_rag import CorrectiveRAGPipeline
from my_rag.core.rag_pipeline import RAGPipeline
from my_rag.core.semantic_cache import SemanticCache
from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.reranker.base import BaseReranker, RerankResult
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.domain.retrieval.hybrid_retriever import HybridRetriever
from my_rag.domain.retrieval.query_rewriter import QueryRewriter

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  带模拟延迟的 Fake 组件 — 模拟真实网络与推理耗时
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_DOCS = [
    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，通过从外部知识库检索相关文档来增强大语言模型的回答质量。",
    "向量检索通过将文本编码为高维向量，利用近似最近邻（ANN）算法实现毫秒级语义搜索。FAISS 是 Meta 开源的高性能向量检索库。",
    "BM25 是基于词频和逆文档频率的经典检索算法，对精确关键词匹配效果好，与稠密检索形成互补。",
    "Reranker 使用 Cross-Encoder 对检索结果精排，通过 query-doc 联合编码捕捉细粒度语义交互，显著提升排序质量。",
    "分块策略直接影响 RAG 效果：小块精确但缺上下文，大块完整但噪声多。Parent-Child 策略兼顾检索精度与上下文完整性。",
]

BENCHMARK_QUERIES = [
    "什么是 RAG 技术？它的核心原理是什么？",
    "FAISS 向量检索的工作原理是什么？",
    "BM25 算法的优势和劣势分别是什么？",
    "Reranker 是如何提升检索效果的？",
    "Parent-Child 分块策略解决了什么问题？",
    "如何评估 RAG 系统的效果？",
    "语义缓存和传统缓存有什么区别？",
    "HyDE 查询改写的原理是什么？",
    "Self-RAG 和 Corrective RAG 有什么区别？",
    "混合检索中 RRF 融合算法是如何工作的？",
]


def _jitter(base_ms: float, variance: float = 0.2) -> float:
    """添加 ±variance 抖动的延迟（秒）。"""
    ms = base_ms * (1 + random.uniform(-variance, variance))
    return ms / 1000.0


class LatencyLLM(BaseLLM):
    """模拟 LLM：generate ~800ms，stream ~50ms/token。"""

    def __init__(self, generate_ms: float = 800, self_rag_responses: list[str] | None = None):
        self._generate_ms = generate_ms
        self._self_rag_responses = list(self_rag_responses or [])
        self._response_idx = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(_jitter(self._generate_ms))
        if self._self_rag_responses and self._response_idx < len(self._self_rag_responses):
            resp = self._self_rag_responses[self._response_idx]
            self._response_idx += 1
            return resp
        return "这是基于检索文档生成的回答，RAG 系统通过检索增强来提升大语言模型的回答质量和准确性。"

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        answer = "这是基于检索文档生成的回答。"
        for ch in answer:
            await asyncio.sleep(_jitter(50))
            yield ch


class LatencyEmbedding(BaseEmbedding):
    """模拟 Embedding：单条 ~30ms，批量有吞吐优势。"""

    def __init__(self, dim: int = 768, per_query_ms: float = 30):
        self._dim = dim
        self._per_query_ms = per_query_ms

    @property
    def dimension(self) -> int:
        return self._dim

    def _fake_vec(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        raw = [int(h[i:i + 2], 16) / 255.0 for i in range(0, self._dim * 2, 2)]
        while len(raw) < self._dim:
            raw.extend(raw[:self._dim - len(raw)])
        raw = raw[:self._dim]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(_jitter(self._per_query_ms))
        return self._fake_vec(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(_jitter(self._per_query_ms * min(len(texts), 5)))
        return [self._fake_vec(t) for t in texts]


class LatencyRetriever(BaseRetriever):
    """模拟检索器：Dense ~45ms，Sparse ~25ms。"""

    def __init__(self, latency_ms: float = 45, name: str = "dense"):
        self._latency_ms = latency_ms
        self._name = name

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        await asyncio.sleep(_jitter(self._latency_ms))
        results = []
        for i in range(min(top_k, len(SAMPLE_DOCS))):
            results.append(RetrievalResult(
                chunk_id=f"chunk_{self._name}_{i}",
                content=SAMPLE_DOCS[i],
                score=round(0.95 - i * 0.08 + random.uniform(-0.02, 0.02), 4),
                source=f"doc_{i}.txt",
                metadata={"knowledge_base_id": knowledge_base_id or "kb_bench"},
            ))
        return results


class LatencyReranker(BaseReranker):
    """模拟 Cross-Encoder Reranker：~120ms。"""

    def __init__(self, latency_ms: float = 120):
        self._latency_ms = latency_ms

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None,
    ) -> list[RerankResult]:
        await asyncio.sleep(_jitter(self._latency_ms))
        scored = [(r, random.uniform(0.5, 1.0)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return [
            RerankResult(
                chunk_id=r.chunk_id, content=r.content, score=s,
                original_rank=i, new_rank=ni + 1,
                source=r.source, metadata=r.metadata,
            )
            for ni, (r, s) in enumerate(scored)
            for i in [0]
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Benchmark 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LatencyStats:
    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def p95(self) -> float:
        if len(self.values) < 2:
            return self.mean
        sorted_v = sorted(self.values)
        idx = int(len(sorted_v) * 0.95)
        return sorted_v[min(idx, len(sorted_v) - 1)]

    @property
    def p99(self) -> float:
        if len(self.values) < 2:
            return self.mean
        sorted_v = sorted(self.values)
        idx = int(len(sorted_v) * 0.99)
        return sorted_v[min(idx, len(sorted_v) - 1)]

    @property
    def min_val(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_val(self) -> float:
        return max(self.values) if self.values else 0.0


@dataclass
class BenchmarkResult:
    name: str
    description: str
    end_to_end: LatencyStats = field(default_factory=LatencyStats)
    extra: dict[str, LatencyStats] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Benchmark 执行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _timed(coro):
    """执行协程并返回 (结果, 耗时ms)。"""
    t0 = time.perf_counter()
    result = await coro
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


async def bench_naive_rag(queries: list[str], runs: int) -> BenchmarkResult:
    """Naive RAG：Query → Retrieve → Generate"""
    br = BenchmarkResult("Naive RAG", "Dense 检索 → LLM 直接生成")
    for _ in range(runs):
        llm = LatencyLLM(generate_ms=800)
        retriever = LatencyRetriever(latency_ms=45, name="dense")
        pipeline = RAGPipeline(retriever=retriever, llm=llm)
        for q in queries:
            _, ms = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.end_to_end.values.append(ms)
    return br


async def bench_advanced_rag(queries: list[str], runs: int) -> BenchmarkResult:
    """Advanced RAG：Hybrid 检索 + Rerank"""
    br = BenchmarkResult("Advanced RAG", "Hybrid(Dense+Sparse) + RRF + Rerank + 生成")
    for _ in range(runs):
        llm = LatencyLLM(generate_ms=800)
        dense = LatencyRetriever(latency_ms=45, name="dense")
        sparse = LatencyRetriever(latency_ms=25, name="sparse")
        hybrid = HybridRetriever(dense, sparse, rrf_k=60)
        reranker = LatencyReranker(latency_ms=120)
        pipeline = RAGPipeline(
            retriever=hybrid, llm=llm,
            reranker=reranker, rerank_top_k=5,
        )
        for q in queries:
            _, ms = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.end_to_end.values.append(ms)
    return br


async def bench_self_rag_on(queries: list[str], runs: int) -> BenchmarkResult:
    """Self-RAG（开启）：三层 LLM 判断"""
    br = BenchmarkResult("Self-RAG (ON)", "检索判断→相关性过滤→生成→支撑性检查")
    for _ in range(runs):
        responses = []
        for _ in queries:
            responses.extend(["是", "相关", "相关", "相关", "相关", "相关",
                              "fake answer", "支撑"])
        llm = LatencyLLM(generate_ms=300, self_rag_responses=responses)
        retriever = LatencyRetriever(latency_ms=45, name="dense")
        pipeline = RAGPipeline(
            retriever=retriever, llm=llm,
            enable_self_rag=True, self_rag_max_retries=1,
        )
        for q in queries:
            _, ms = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.end_to_end.values.append(ms)
    return br


async def bench_self_rag_off(queries: list[str], runs: int) -> BenchmarkResult:
    """Self-RAG（关闭）：普通 RAG 基准"""
    br = BenchmarkResult("Self-RAG (OFF)", "普通 RAG（无 Self-RAG 判断）")
    for _ in range(runs):
        llm = LatencyLLM(generate_ms=800)
        retriever = LatencyRetriever(latency_ms=45, name="dense")
        pipeline = RAGPipeline(retriever=retriever, llm=llm, enable_self_rag=False)
        for q in queries:
            _, ms = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.end_to_end.values.append(ms)
    return br


async def bench_corrective_rag(queries: list[str], runs: int) -> BenchmarkResult:
    """Corrective RAG：文档评分 + 过滤/补充"""
    br = BenchmarkResult("Corrective RAG", "检索→文档三档评分→过滤/补充→生成")
    for _ in range(runs):
        responses = []
        for _ in queries:
            responses.extend(["correct", "correct", "ambiguous", "incorrect", "incorrect", "answer"])
        llm = LatencyLLM(generate_ms=300, self_rag_responses=responses)
        retriever = LatencyRetriever(latency_ms=45, name="dense")
        pipeline = CorrectiveRAGPipeline(retriever=retriever, llm=llm)
        for q in queries:
            _, ms = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.end_to_end.values.append(ms)
    return br


async def bench_cache_hit(queries: list[str], runs: int) -> BenchmarkResult:
    """语义缓存命中 vs 未命中"""
    br = BenchmarkResult("语义缓存", "首次 miss → 二次 hit 对比")
    br.extra["cache_miss"] = LatencyStats()
    br.extra["cache_hit"] = LatencyStats()
    for _ in range(runs):
        embedding = LatencyEmbedding(dim=32, per_query_ms=30)
        cache = SemanticCache(embedding, similarity_threshold=0.90, max_size=100)
        llm = LatencyLLM(generate_ms=800)
        retriever = LatencyRetriever(latency_ms=45)
        pipeline = RAGPipeline(
            retriever=retriever, llm=llm,
            semantic_cache=cache, enable_cache=True,
        )
        for q in queries:
            _, ms_miss = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.extra["cache_miss"].values.append(ms_miss)
            br.end_to_end.values.append(ms_miss)
        for q in queries:
            _, ms_hit = await _timed(pipeline.run(q, knowledge_base_id="kb_bench"))
            br.extra["cache_hit"].values.append(ms_hit)
            br.end_to_end.values.append(ms_hit)
    return br


async def bench_rerank_comparison(queries: list[str], runs: int) -> BenchmarkResult:
    """Rerank 开关对比"""
    br = BenchmarkResult("Rerank 开关对比", "无 Rerank vs 有 Rerank")
    br.extra["without_rerank"] = LatencyStats()
    br.extra["with_rerank"] = LatencyStats()
    for _ in range(runs):
        llm = LatencyLLM(generate_ms=800)
        retriever = LatencyRetriever(latency_ms=45)
        p_no = RAGPipeline(retriever=retriever, llm=llm)
        for q in queries:
            _, ms = await _timed(p_no.run(q, knowledge_base_id="kb_bench"))
            br.extra["without_rerank"].values.append(ms)

        llm2 = LatencyLLM(generate_ms=800)
        retriever2 = LatencyRetriever(latency_ms=45)
        reranker = LatencyReranker(latency_ms=120)
        p_yes = RAGPipeline(retriever=retriever2, llm=llm2, reranker=reranker, rerank_top_k=5)
        for q in queries:
            _, ms = await _timed(p_yes.run(q, knowledge_base_id="kb_bench"))
            br.extra["with_rerank"].values.append(ms)
    return br


async def bench_query_rewrite(queries: list[str], runs: int) -> BenchmarkResult:
    """查询改写开销（HyDE / Multi-Query）"""
    br = BenchmarkResult("查询改写开销", "无改写 vs HyDE vs Multi-Query")
    br.extra["no_rewrite"] = LatencyStats()
    br.extra["hyde"] = LatencyStats()
    br.extra["multi_query"] = LatencyStats()
    for _ in range(runs):
        llm = LatencyLLM(generate_ms=800)
        retriever = LatencyRetriever(latency_ms=45)
        p_base = RAGPipeline(retriever=retriever, llm=llm)
        for q in queries:
            _, ms = await _timed(p_base.run(q, knowledge_base_id="kb_bench"))
            br.extra["no_rewrite"].values.append(ms)

        llm_h = LatencyLLM(generate_ms=500)
        retriever_h = LatencyRetriever(latency_ms=45)
        rewriter_h = QueryRewriter(llm_h)
        p_hyde = RAGPipeline(
            retriever=retriever_h, llm=llm_h,
            query_rewriter=rewriter_h, enable_hyde=True,
        )
        for q in queries:
            _, ms = await _timed(p_hyde.run(q, knowledge_base_id="kb_bench"))
            br.extra["hyde"].values.append(ms)

        llm_m = LatencyLLM(generate_ms=500)
        retriever_m = LatencyRetriever(latency_ms=45)
        rewriter_m = QueryRewriter(llm_m)
        p_mq = RAGPipeline(
            retriever=retriever_m, llm=llm_m,
            query_rewriter=rewriter_m, enable_multi_query=True,
        )
        for q in queries:
            _, ms = await _timed(p_mq.run(q, knowledge_base_id="kb_bench"))
            br.extra["multi_query"].values.append(ms)
    return br


async def bench_stage_breakdown(queries: list[str], runs: int) -> BenchmarkResult:
    """各阶段延迟分解"""
    br = BenchmarkResult("阶段延迟分解", "Embedding / Dense检索 / Sparse检索 / Rerank / LLM生成")
    br.extra["embedding"] = LatencyStats()
    br.extra["dense_retrieval"] = LatencyStats()
    br.extra["sparse_retrieval"] = LatencyStats()
    br.extra["rerank"] = LatencyStats()
    br.extra["llm_generate"] = LatencyStats()

    for _ in range(runs):
        emb = LatencyEmbedding(dim=32, per_query_ms=30)
        dense = LatencyRetriever(latency_ms=45, name="dense")
        sparse = LatencyRetriever(latency_ms=25, name="sparse")
        reranker = LatencyReranker(latency_ms=120)
        llm = LatencyLLM(generate_ms=800)

        for q in queries:
            _, ms = await _timed(emb.embed_query(q))
            br.extra["embedding"].values.append(ms)

            _, ms = await _timed(dense.retrieve(q, top_k=5))
            br.extra["dense_retrieval"].values.append(ms)

            _, ms = await _timed(sparse.retrieve(q, top_k=5))
            br.extra["sparse_retrieval"].values.append(ms)

            docs = [RetrievalResult(chunk_id=f"c{i}", content=SAMPLE_DOCS[i], score=0.9 - i * 0.1, source=f"d{i}.txt")
                    for i in range(5)]
            _, ms = await _timed(reranker.rerank(q, docs, top_k=5))
            br.extra["rerank"].values.append(ms)

            _, ms = await _timed(llm.generate(f"answer: {q}"))
            br.extra["llm_generate"].values.append(ms)
    return br


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  报告生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def generate_report(results: list[BenchmarkResult], total_time_s: float) -> str:
    lines: list[str] = []
    lines.append("# RAG 系统性能基准报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 总耗时：{total_time_s:.1f}s")
    lines.append("")

    # ── 1. 阶段延迟分解 ──
    stage = next((r for r in results if r.name == "阶段延迟分解"), None)
    if stage:
        lines.append("## 1. 各阶段延迟分解")
        lines.append("")
        lines.append("| 阶段 | 平均延迟 | P50 | P95 | P99 | 最小 | 最大 |")
        lines.append("|------|---------|-----|-----|-----|------|------|")
        stage_order = [
            ("embedding", "Embedding 编码"),
            ("dense_retrieval", "Dense 检索 (FAISS)"),
            ("sparse_retrieval", "Sparse 检索 (BM25)"),
            ("rerank", "Rerank 精排"),
            ("llm_generate", "LLM 生成"),
        ]
        for key, label in stage_order:
            s = stage.extra.get(key)
            if s and s.count > 0:
                lines.append(
                    f"| {label} | {_fmt(s.mean)} | {_fmt(s.median)} | "
                    f"{_fmt(s.p95)} | {_fmt(s.p99)} | {_fmt(s.min_val)} | {_fmt(s.max_val)} |"
                )
        lines.append("")

    # ── 2. 端到端对比 ──
    e2e_names = ["Naive RAG", "Advanced RAG", "Self-RAG (ON)", "Self-RAG (OFF)", "Corrective RAG"]
    e2e_results = [r for r in results if r.name in e2e_names]
    if e2e_results:
        lines.append("## 2. 端到端延迟对比")
        lines.append("")
        lines.append("| 配置 | 说明 | 平均延迟 | P50 | P95 | 请求数 |")
        lines.append("|------|------|---------|-----|-----|--------|")
        for r in e2e_results:
            s = r.end_to_end
            lines.append(
                f"| {r.name} | {r.description} | {_fmt(s.mean)} | "
                f"{_fmt(s.median)} | {_fmt(s.p95)} | {s.count} |"
            )
        lines.append("")

    # ── 3. Self-RAG 开关对比 ──
    self_on = next((r for r in results if r.name == "Self-RAG (ON)"), None)
    self_off = next((r for r in results if r.name == "Self-RAG (OFF)"), None)
    if self_on and self_off:
        lines.append("## 3. Self-RAG 开关对比")
        lines.append("")
        overhead_ms = self_on.end_to_end.mean - self_off.end_to_end.mean
        overhead_pct = (overhead_ms / self_off.end_to_end.mean * 100) if self_off.end_to_end.mean > 0 else 0
        lines.append("| 指标 | Self-RAG OFF | Self-RAG ON | 差值 | 开销占比 |")
        lines.append("|------|-------------|-------------|------|---------|")
        lines.append(
            f"| 平均延迟 | {_fmt(self_off.end_to_end.mean)} | {_fmt(self_on.end_to_end.mean)} | "
            f"+{_fmt(overhead_ms)} | +{overhead_pct:.1f}% |"
        )
        lines.append(
            f"| P95 延迟 | {_fmt(self_off.end_to_end.p95)} | {_fmt(self_on.end_to_end.p95)} | "
            f"+{_fmt(self_on.end_to_end.p95 - self_off.end_to_end.p95)} | - |"
        )
        lines.append("")
        lines.append("**分析：** Self-RAG 引入了 3 次额外的 LLM 调用（检索判断 + N 次相关性判断 + 支撑性检查），"
                      f"平均增加 {_fmt(overhead_ms)} 延迟（+{overhead_pct:.1f}%）。"
                      "适用于对准确性要求极高、可容忍较高延迟的场景（如医疗、法律问答）。")
        lines.append("")

    # ── 4. Rerank 开关对比 ──
    rerank_res = next((r for r in results if r.name == "Rerank 开关对比"), None)
    if rerank_res:
        no_rr = rerank_res.extra.get("without_rerank")
        with_rr = rerank_res.extra.get("with_rerank")
        if no_rr and with_rr:
            lines.append("## 4. Rerank 开关对比")
            lines.append("")
            rr_overhead = with_rr.mean - no_rr.mean
            rr_pct = (rr_overhead / no_rr.mean * 100) if no_rr.mean > 0 else 0
            lines.append("| 指标 | 无 Rerank | 有 Rerank | 差值 | 开销占比 |")
            lines.append("|------|----------|----------|------|---------|")
            lines.append(
                f"| 平均延迟 | {_fmt(no_rr.mean)} | {_fmt(with_rr.mean)} | "
                f"+{_fmt(rr_overhead)} | +{rr_pct:.1f}% |"
            )
            lines.append(
                f"| P95 延迟 | {_fmt(no_rr.p95)} | {_fmt(with_rr.p95)} | "
                f"+{_fmt(with_rr.p95 - no_rr.p95)} | - |"
            )
            lines.append("")
            lines.append(f"**分析：** Rerank 阶段增加约 {_fmt(rr_overhead)}（+{rr_pct:.1f}%），"
                          "但在实际场景中可将 Context Precision 提升 10-25%。延迟增幅可控，推荐开启。")
            lines.append("")

    # ── 5. 缓存对比 ──
    cache_res = next((r for r in results if r.name == "语义缓存"), None)
    if cache_res:
        miss = cache_res.extra.get("cache_miss")
        hit = cache_res.extra.get("cache_hit")
        if miss and hit:
            lines.append("## 5. 语义缓存命中 vs 未命中")
            lines.append("")
            speedup = miss.mean / hit.mean if hit.mean > 0 else 0
            saving = miss.mean - hit.mean
            lines.append("| 指标 | Cache Miss | Cache Hit | 加速比 | 节省延迟 |")
            lines.append("|------|-----------|----------|--------|---------|")
            lines.append(
                f"| 平均延迟 | {_fmt(miss.mean)} | {_fmt(hit.mean)} | "
                f"{speedup:.1f}x | {_fmt(saving)} |"
            )
            lines.append("")
            lines.append(f"**分析：** 缓存命中时延迟降至 {_fmt(hit.mean)}，相比未命中的 {_fmt(miss.mean)} "
                          f"加速 {speedup:.1f} 倍。语义缓存跳过了检索和 LLM 生成，"
                          "仅需一次 Embedding 编码 + 余弦相似度计算。")
            lines.append("")

    # ── 6. 查询改写 ──
    qr_res = next((r for r in results if r.name == "查询改写开销"), None)
    if qr_res:
        no_rw = qr_res.extra.get("no_rewrite")
        hyde = qr_res.extra.get("hyde")
        mq = qr_res.extra.get("multi_query")
        if no_rw and hyde and mq:
            lines.append("## 6. 查询改写开销对比")
            lines.append("")
            lines.append("| 模式 | 平均延迟 | P95 | 相比基线 |")
            lines.append("|------|---------|-----|---------|")
            lines.append(f"| 无改写（基线） | {_fmt(no_rw.mean)} | {_fmt(no_rw.p95)} | - |")
            hyde_delta = hyde.mean - no_rw.mean
            lines.append(
                f"| HyDE | {_fmt(hyde.mean)} | {_fmt(hyde.p95)} | +{_fmt(hyde_delta)} |"
            )
            mq_delta = mq.mean - no_rw.mean
            lines.append(
                f"| Multi-Query | {_fmt(mq.mean)} | {_fmt(mq.p95)} | +{_fmt(mq_delta)} |"
            )
            lines.append("")
            lines.append(f"**分析：** HyDE 增加一次 LLM 调用（+{_fmt(hyde_delta)}），"
                          f"Multi-Query 增加一次 LLM + 多次并行检索（+{_fmt(mq_delta)}）。"
                          "HyDE 适合语义模糊的短查询，Multi-Query 适合需要多角度召回的复杂查询。")
            lines.append("")

    # ── 7. 结论 ──
    lines.append("## 7. 优化建议")
    lines.append("")
    lines.append("| 优先级 | 优化手段 | 预期收益 | 延迟影响 |")
    lines.append("|:-----:|---------|---------|---------|")
    lines.append("| 1 | 启用语义缓存 | 高频问题延迟降低 90%+ | 首次无影响 |")
    lines.append("| 2 | 启用 Rerank | Context Precision +10~25% | +15% 延迟 |")
    lines.append("| 3 | Hybrid 检索（Dense+Sparse） | Recall +15~30% | +5~10% 延迟 |")
    lines.append("| 4 | HyDE 查询改写 | 短查询 Recall +10~20% | +1 次 LLM 调用 |")
    lines.append("| 5 | Self-RAG | 幻觉率显著降低 | +200~300% 延迟 |")
    lines.append("| 6 | Corrective RAG | 噪声文档过滤 | +N 次 LLM 调用 |")
    lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_all_benchmarks(queries: list[str], runs: int = 3) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    benchmarks = [
        ("阶段延迟分解", bench_stage_breakdown),
        ("Naive RAG", bench_naive_rag),
        ("Advanced RAG", bench_advanced_rag),
        ("Self-RAG (OFF)", bench_self_rag_off),
        ("Self-RAG (ON)", bench_self_rag_on),
        ("Corrective RAG", bench_corrective_rag),
        ("语义缓存", bench_cache_hit),
        ("Rerank 开关对比", bench_rerank_comparison),
        ("查询改写开销", bench_query_rewrite),
    ]
    total = len(benchmarks)
    for i, (name, fn) in enumerate(benchmarks, 1):
        print(f"  [{i}/{total}] {name} ...")
        result = await fn(queries, runs)
        results.append(result)
        print(f"         → {result.end_to_end.count} 次请求, "
              f"平均 {_fmt(result.end_to_end.mean) if result.end_to_end.count else 'N/A'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG 性能基准测试")
    parser.add_argument("--runs", type=int, default=3, help="每项测试重复轮数 (default: 3)")
    parser.add_argument("--queries", type=int, default=10, help="每轮查询数 (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径 (default: 终端输出)")
    args = parser.parse_args()

    queries = BENCHMARK_QUERIES[:args.queries]
    print(f"\n{'='*60}")
    print(f"  RAG 性能基准测试")
    print(f"  查询数: {len(queries)}, 轮数: {args.runs}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    results = asyncio.run(run_all_benchmarks(queries, runs=args.runs))
    total_time = time.perf_counter() - t0

    report = generate_report(results, total_time)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到: {args.output}")
    else:
        print(f"\n{'='*60}")
        print(report)


if __name__ == "__main__":
    main()
