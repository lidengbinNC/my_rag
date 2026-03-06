"""Tests for RRF (Reciprocal Rank Fusion) algorithm and HybridRetriever."""

import pytest

from my_rag.domain.retrieval.base import RetrievalResult
from my_rag.domain.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
)
from tests.conftest import FakeRetriever


def _make(chunk_id: str, score: float, source: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, content=f"content of {chunk_id}", score=score, source=source
    )


# ── reciprocal_rank_fusion 核心算法 ──────────────────────────────────


class TestRRF:
    def test_single_list_ranking(self):
        """单列表 RRF 应按倒数排名排序。"""
        ranked = [_make("a", 0.9), _make("b", 0.8), _make("c", 0.7)]
        fused = reciprocal_rank_fusion([ranked], k=60)

        assert [r.chunk_id for r in fused] == ["a", "b", "c"]
        assert fused[0].score == pytest.approx(1.0 / (60 + 1))
        assert fused[1].score == pytest.approx(1.0 / (60 + 2))

    def test_two_lists_fusion(self):
        """两个列表中共同出现的文档应获得更高的融合分数。"""
        list1 = [_make("a", 0.9), _make("b", 0.7)]
        list2 = [_make("b", 0.8), _make("c", 0.6)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        scores = {r.chunk_id: r.score for r in fused}
        # b appears in both lists → highest fused score
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_disjoint_lists(self):
        """完全不重叠的两个列表也能正确融合。"""
        list1 = [_make("a", 0.9)]
        list2 = [_make("b", 0.8)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        assert len(fused) == 2
        assert fused[0].score == fused[1].score  # both rank-1 in their list

    def test_empty_lists(self):
        """空列表返回空结果。"""
        assert reciprocal_rank_fusion([[], []], k=60) == []
        assert reciprocal_rank_fusion([], k=60) == []

    def test_k_parameter_affects_smoothing(self):
        """更小的 k 值让排名差异更大。"""
        ranked = [_make("a", 0.9), _make("b", 0.7)]
        fused_small_k = reciprocal_rank_fusion([ranked], k=1)
        fused_large_k = reciprocal_rank_fusion([ranked], k=100)

        gap_small = fused_small_k[0].score - fused_small_k[1].score
        gap_large = fused_large_k[0].score - fused_large_k[1].score
        assert gap_small > gap_large

    def test_preserves_content_and_metadata(self):
        """融合后保留原始内容和元数据。"""
        original = _make("x", 0.5, source="doc.txt")
        original.metadata = {"key": "value"}
        fused = reciprocal_rank_fusion([[original]], k=60)
        assert fused[0].content == original.content
        assert fused[0].source == "doc.txt"
        assert fused[0].metadata == {"key": "value"}


# ── HybridRetriever 集成 ─────────────────────────────────────────────


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_combines_dense_and_sparse(self):
        """混合检索器并发调用 dense 和 sparse 并融合结果。"""
        dense = FakeRetriever([_make("a", 0.9), _make("b", 0.7)])
        sparse = FakeRetriever([_make("b", 0.8), _make("c", 0.6)])
        hybrid = HybridRetriever(dense, sparse, rrf_k=60)

        results = await hybrid.retrieve("test query", top_k=3)
        ids = [r.chunk_id for r in results]

        assert "b" in ids  # appears in both → should rank high
        assert dense.call_count == 1
        assert sparse.call_count == 1

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        """top_k 截断融合结果。"""
        dense = FakeRetriever([_make(f"d{i}", 0.9 - i * 0.1) for i in range(5)])
        sparse = FakeRetriever([_make(f"s{i}", 0.8 - i * 0.1) for i in range(5)])
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve("q", top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_retrievers(self):
        """两个空检索器返回空结果。"""
        hybrid = HybridRetriever(FakeRetriever([]), FakeRetriever([]))
        results = await hybrid.retrieve("q", top_k=5)
        assert results == []
