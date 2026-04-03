"""Shared fixtures for all tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.reranker.base import BaseReranker, RerankResult
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult


# ── Fake LLM ────────────────────────────────────────────────────────


class FakeLLM(BaseLLM):
    """Controllable fake LLM for testing.

    Supply ``responses`` (a list of strings); each call to ``generate``
    pops the first item. If the list is exhausted it falls back to
    ``default_response``.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "fake answer",
    ):
        self.responses = list(responses or [])
        self.default_response = default_response
        self.call_history: list[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.call_history.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return self.default_response

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        self.call_history.append(prompt)
        text = self.responses.pop(0) if self.responses else self.default_response
        for ch in text:
            yield ch


# ── Fake Embedding ──────────────────────────────────────────────────


class FakeEmbedding(BaseEmbedding):
    """Returns deterministic embeddings based on text hash."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return True

    def _embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        raw = [int(h[i : i + 2], 16) / 255.0 for i in range(0, self._dim * 2, 2)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_documents_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        dense = await self.embed_documents(texts)
        sparse = [{i: value for i, value in enumerate(vec) if value > 0} for vec in dense]
        return dense, sparse

    async def embed_query_hybrid(self, text: str) -> tuple[list[float], dict[int, float]]:
        dense = await self.embed_query(text)
        sparse = {i: value for i, value in enumerate(dense) if value > 0}
        return dense, sparse


# ── Fake Retriever ──────────────────────────────────────────────────


class FakeRetriever(BaseRetriever):
    """Returns pre-loaded results regardless of query."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = results or []
        self.call_count = 0

    async def retrieve(
        self, query: str, top_k: int = 5, knowledge_base_id: str | None = None
    ) -> list[RetrievalResult]:
        self.call_count += 1
        return self.results[:top_k]


# ── Fake Reranker ───────────────────────────────────────────────────


class FakeReranker(BaseReranker):
    """Reverses the input order to simulate reranking."""

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        reversed_results = list(reversed(results))
        out = [
            RerankResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=1.0 / (i + 1),
                original_rank=len(results) - i,
                new_rank=i + 1,
                source=r.source,
                metadata=r.metadata,
            )
            for i, r in enumerate(reversed_results)
        ]
        if top_k:
            out = out[:top_k]
        return out


# ── Convenience fixtures ────────────────────────────────────────────


def make_retrieval_results(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            content=f"这是第 {i} 段测试文档的内容。",
            score=1.0 - i * 0.1,
            source=f"doc_{i}.txt",
            metadata={"knowledge_base_id": "kb_1"},
        )
        for i in range(n)
    ]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever(results=make_retrieval_results(5))


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    return make_retrieval_results(5)
