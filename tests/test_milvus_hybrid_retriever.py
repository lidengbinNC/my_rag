from __future__ import annotations

import pytest

from my_rag.config.settings import settings
from my_rag.core import dependencies as deps
from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.retrieval.base import RetrievalResult
from my_rag.domain.retrieval.dense_retriever import DenseRetriever
from my_rag.domain.retrieval.milvus_hybrid_retriever import MilvusHybridRetriever
from my_rag.infrastructure.vector_store.base import BaseVectorStore, VectorSearchResult
from tests.conftest import FakeEmbedding


class FakeHybridVectorStore(BaseVectorStore):
    def __init__(self, results: list[VectorSearchResult] | None = None):
        self._results = results or []
        self.last_hybrid_kwargs: dict | None = None

    @property
    def supports_hybrid(self) -> bool:
        return True

    async def add(self, ids, embeddings, texts, metadatas=None) -> None:
        return None

    async def add_hybrid(
        self, ids, embeddings, sparse_embeddings, texts, metadatas=None
    ) -> None:
        return None

    async def search(self, query_embedding, top_k=5, filter_metadata=None):
        return []

    async def search_hybrid(
        self,
        query_embedding,
        query_sparse_embedding,
        top_k=5,
        filter_metadata=None,
        dense_weight=0.5,
        sparse_weight=0.5,
        ranker="weighted",
        candidate_limit=None,
        rrf_k=60,
    ) -> list[VectorSearchResult]:
        self.last_hybrid_kwargs = {
            "query_embedding": query_embedding,
            "query_sparse_embedding": query_sparse_embedding,
            "top_k": top_k,
            "filter_metadata": filter_metadata,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "ranker": ranker,
            "candidate_limit": candidate_limit,
            "rrf_k": rrf_k,
        }
        return self._results[:top_k]

    async def delete(self, ids) -> None:
        return None

    async def delete_by_metadata(self, key, value) -> None:
        return None

    def count(self) -> int:
        return len(self._results)


class DenseOnlyEmbedding(BaseEmbedding):
    @property
    def dimension(self) -> int:
        return 4

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_milvus_hybrid_retriever_calls_vector_store_hybrid():
    embedding = FakeEmbedding()
    store = FakeHybridVectorStore(results=[
        VectorSearchResult(
            chunk_id="chunk-1",
            score=0.9,
            content="test content",
            metadata={"source": "doc.txt", "knowledge_base_id": "kb-1"},
        )
    ])
    retriever = MilvusHybridRetriever(
        embedding=embedding,
        vector_store=store,
        ranker="weighted",
        dense_weight=0.7,
        sparse_weight=0.3,
        candidate_limit=12,
        rrf_k=80,
    )

    results = await retriever.retrieve("hello world", top_k=1, knowledge_base_id="kb-1")

    assert [r.chunk_id for r in results] == ["chunk-1"]
    assert store.last_hybrid_kwargs is not None
    assert store.last_hybrid_kwargs["top_k"] == 1
    assert store.last_hybrid_kwargs["filter_metadata"] == {"knowledge_base_id": "kb-1"}
    assert store.last_hybrid_kwargs["dense_weight"] == pytest.approx(0.7)
    assert store.last_hybrid_kwargs["sparse_weight"] == pytest.approx(0.3)
    assert store.last_hybrid_kwargs["ranker"] == "weighted"
    assert store.last_hybrid_kwargs["candidate_limit"] == 12
    assert store.last_hybrid_kwargs["rrf_k"] == 80
    assert store.last_hybrid_kwargs["query_sparse_embedding"]


def test_get_retriever_prefers_milvus_hybrid(monkeypatch: pytest.MonkeyPatch):
    hybrid_store = FakeHybridVectorStore()

    monkeypatch.setattr(deps, "_retriever", None)
    monkeypatch.setattr(deps, "_embedding", FakeEmbedding())
    monkeypatch.setattr(deps, "_vector_store", hybrid_store)
    monkeypatch.setattr(settings.vector_store, "provider", "milvus", raising=False)
    monkeypatch.setattr(settings.retrieval, "enable_milvus_hybrid", True, raising=False)
    monkeypatch.setattr(settings.retrieval, "hybrid_ranker", "weighted", raising=False)
    monkeypatch.setattr(settings.retrieval, "hybrid_dense_weight", 0.6, raising=False)
    monkeypatch.setattr(settings.retrieval, "hybrid_sparse_weight", 0.4, raising=False)
    monkeypatch.setattr(settings.retrieval, "hybrid_candidate_limit", 16, raising=False)
    monkeypatch.setattr(settings.retrieval, "rrf_k", 60, raising=False)

    retriever = deps.get_retriever()

    assert isinstance(retriever, MilvusHybridRetriever)


def test_get_retriever_falls_back_to_dense_when_sparse_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    hybrid_store = FakeHybridVectorStore()

    monkeypatch.setattr(deps, "_retriever", None)
    monkeypatch.setattr(deps, "_embedding", DenseOnlyEmbedding())
    monkeypatch.setattr(deps, "_vector_store", hybrid_store)
    monkeypatch.setattr(settings.vector_store, "provider", "milvus", raising=False)
    monkeypatch.setattr(settings.retrieval, "enable_milvus_hybrid", True, raising=False)

    retriever = deps.get_retriever()

    assert isinstance(retriever, DenseRetriever)
