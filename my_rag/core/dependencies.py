"""
依赖注入 — 组件初始化与单例管理

面试考点：
- 模块级单例：利用 Python 模块的天然单例特性管理全局组件
- 对比 Java 的 Spring IoC 容器：Python 更轻量，用工厂 + 全局变量即可
- 延迟初始化（Lazy Init）：首次访问时才创建，避免启动时加载全部重量级模型
- 组件依赖链：Embedding → VectorStore → DenseRetriever + SparseRetriever → HybridRetriever
                                                                              ↓
                                         LLM → QueryRewriter + SemanticCache → RAGPipeline
"""

from my_rag.config.settings import settings
from my_rag.core.rag_pipeline import RAGPipeline
from my_rag.core.semantic_cache import SemanticCache
from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.retrieval.base import BaseRetriever
from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from my_rag.infrastructure.notification.base import BaseNotifier
from my_rag.infrastructure.vector_store.base import BaseVectorStore

_embedding: BaseEmbedding | None = None
_vector_store: BaseVectorStore | None = None
_retriever: BaseRetriever | None = None
_llm: BaseLLM | None = None
_query_rewriter: QueryRewriter | None = None
_semantic_cache: SemanticCache | None = None
_rag_pipeline: RAGPipeline | None = None
_dingtalk_notifier: BaseNotifier | None = None


def get_embedding() -> BaseEmbedding:
    global _embedding
    if _embedding is None:
        from my_rag.domain.embedding.factory import EmbeddingFactory
        _embedding = EmbeddingFactory.create(
            provider=settings.embedding.provider,
            model=settings.embedding.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            dimension=settings.embedding.dimension,
        )
    return _embedding


def get_vector_store() -> BaseVectorStore:
    global _vector_store
    if _vector_store is None:
        from my_rag.infrastructure.vector_store.faiss_store import FAISSVectorStore
        _vector_store = FAISSVectorStore(
            dimension=get_embedding().dimension,
            persist_dir=str(settings.data_dir / "faiss_index"),
        )
    return _vector_store


def get_sparse_retriever():
    from my_rag.domain.retrieval.sparse_retriever import SparseRetriever
    if not hasattr(get_sparse_retriever, "_instance"):
        get_sparse_retriever._instance = SparseRetriever()
    return get_sparse_retriever._instance


def get_retriever() -> BaseRetriever:
    global _retriever
    if _retriever is None:
        from my_rag.domain.retrieval.dense_retriever import DenseRetriever
        from my_rag.domain.retrieval.hybrid_retriever import HybridRetriever

        dense = DenseRetriever(embedding=get_embedding(), vector_store=get_vector_store())
        sparse = get_sparse_retriever()

        _retriever = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            rrf_k=settings.retrieval.rrf_k,
        )
    return _retriever


def get_llm() -> BaseLLM:
    global _llm
    if _llm is None:
        from my_rag.domain.llm.factory import LLMFactory
        _llm = LLMFactory.create(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
    return _llm


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(llm=get_llm())
    return _query_rewriter


def get_semantic_cache() -> SemanticCache:
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache(
            embedding=get_embedding(),
            similarity_threshold=settings.retrieval.cache_similarity_threshold,
            max_size=500,
            ttl_seconds=settings.retrieval.cache_ttl_seconds,
        )
    return _semantic_cache


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            retriever=get_retriever(),
            llm=get_llm(),
            query_rewriter=get_query_rewriter(),
            semantic_cache=get_semantic_cache(),
            enable_hyde=settings.retrieval.enable_hyde,
            enable_multi_query=settings.retrieval.enable_multi_query,
            enable_cache=settings.retrieval.enable_cache,
        )
    return _rag_pipeline


def get_dingtalk_notifier() -> BaseNotifier | None:
    """获取钉钉通知器（未启用时返回 None）"""
    global _dingtalk_notifier
    if _dingtalk_notifier is None:
        if not settings.dingtalk.enabled or not settings.dingtalk.webhook_url:
            return None
        from my_rag.infrastructure.notification.dingtalk_notifier import DingTalkNotifier
        _dingtalk_notifier = DingTalkNotifier(
            webhook_url=settings.dingtalk.webhook_url,
            secret=settings.dingtalk.secret,
        )
    return _dingtalk_notifier
