"""
依赖注入 — 组件初始化与单例管理

面试考点：
- 模块级单例：利用 Python 模块的天然单例特性管理全局组件
- 对比 Java 的 Spring IoC 容器：Python 更轻量，用工厂 + 全局变量即可
- 延迟初始化（Lazy Init）：首次访问时才创建，避免启动时加载全部重量级模型
- 组件依赖链：Embedding → VectorStore(FAISS/Milvus) → DenseRetriever + SparseRetriever → HybridRetriever
                                                                                              ↓
                                             LLM → QueryRewriter + SemanticCache → RAGPipeline
- VectorStore 切换：通过 VECTOR_STORE_PROVIDER 环境变量，零代码改动切换 FAISS ↔ Milvus
"""

from my_rag.config.settings import settings
from my_rag.core.rag_pipeline import RAGPipeline
from my_rag.utils.logger import get_logger
from my_rag.core.semantic_cache import SemanticCache
from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.reranker.base import BaseReranker
from my_rag.domain.retrieval.base import BaseRetriever
from my_rag.domain.retrieval.query_rewriter import QueryRewriter
from my_rag.infrastructure.notification.base import BaseNotifier
from my_rag.infrastructure.vector_store.base import BaseVectorStore

logger = get_logger(__name__)

_embedding: BaseEmbedding | None = None
_vector_store: BaseVectorStore | None = None
_retriever: BaseRetriever | None = None
_llm: BaseLLM | None = None
_query_rewriter: QueryRewriter | None = None
_semantic_cache: SemanticCache | None = None
_reranker: BaseReranker | None = None
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
    """
    获取向量存储实例（支持 FAISS / Milvus 动态切换）

    通过 VECTOR_STORE_PROVIDER 环境变量控制：
      VECTOR_STORE_PROVIDER=faiss   → 本地 FAISS（开发/测试）
      VECTOR_STORE_PROVIDER=milvus  → Milvus 服务（生产）
    """
    global _vector_store
    if _vector_store is None:
        from my_rag.infrastructure.vector_store.factory import VectorStoreFactory

        provider = settings.vector_store.provider
        vs_cfg = settings.vector_store

        if provider == "milvus":
            _vector_store = VectorStoreFactory.create(
                provider="milvus",
                dimension=get_embedding().dimension,
                collection_name=vs_cfg.milvus_collection,
                host=vs_cfg.milvus_host,
                port=vs_cfg.milvus_port,
                user=vs_cfg.milvus_user,
                password=vs_cfg.milvus_password,
                db_name=vs_cfg.milvus_db_name,
                uri=vs_cfg.milvus_uri,
                token=vs_cfg.milvus_token,
                index_type=vs_cfg.milvus_index_type,
                metric_type=vs_cfg.milvus_metric_type,
                hnsw_m=vs_cfg.milvus_hnsw_m,
                hnsw_ef_construction=vs_cfg.milvus_hnsw_ef_construction,
                hnsw_ef_search=vs_cfg.milvus_hnsw_ef_search,
            )
        else:
            _vector_store = VectorStoreFactory.create(
                provider="faiss",
                dimension=get_embedding().dimension,
                persist_dir=str(settings.data_dir / "faiss_index"),
            )

        logger.info("vector_store_initialized", provider=provider)
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


def get_reranker() -> BaseReranker | None:
    """获取 Reranker（未启用时返回 None）"""
    global _reranker
    if _reranker is None:
        if not settings.retrieval.enable_rerank:
            return None
        from my_rag.domain.reranker.factory import RerankerFactory
        _reranker = RerankerFactory.create(
            provider=settings.retrieval.rerank_provider,
            model=settings.retrieval.rerank_model,
            score_threshold=settings.retrieval.rerank_score_threshold,
            llm=get_llm() if settings.retrieval.rerank_provider == "llm" else None,
        )
    return _reranker


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            retriever=get_retriever(),
            llm=get_llm(),
            query_rewriter=get_query_rewriter(),
            semantic_cache=get_semantic_cache(),
            reranker=get_reranker(),
            enable_hyde=settings.retrieval.enable_hyde,
            enable_multi_query=settings.retrieval.enable_multi_query,
            enable_cache=settings.retrieval.enable_cache,
            rerank_top_k=settings.retrieval.rerank_top_k,
        )
    return _rag_pipeline


async def shutdown_vector_store() -> None:
    """
    优雅关闭向量存储连接（在 FastAPI lifespan 退出时调用）

    对 FAISS：无需操作（内存数据已在写入时持久化）
    对 Milvus：释放 Collection 内存占用，断开 gRPC 连接
    """
    global _vector_store
    if _vector_store is not None and settings.vector_store.provider == "milvus":
        from my_rag.infrastructure.vector_store.milvus_store import MilvusVectorStore
        if isinstance(_vector_store, MilvusVectorStore):
            await _vector_store.close()
            logger.info("milvus_connection_closed")


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
