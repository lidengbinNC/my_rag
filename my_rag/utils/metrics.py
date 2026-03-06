"""
Prometheus 指标收集

面试考点：
- Prometheus 的四种指标类型：Counter / Gauge / Histogram / Summary
  - Counter：只增不减（如请求总数、错误数）
  - Gauge：可增可减（如当前连接数、缓存大小）
  - Histogram：分布统计（如请求耗时的 p50/p95/p99）
  - Summary：类似 Histogram 但在客户端计算分位数
- 指标命名规范：{namespace}_{subsystem}_{name}_{unit}
- 标签（label）设计：不宜过多（高基数问题），关键维度即可
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST


# === Request Metrics ===
REQUEST_COUNT = Counter(
    "rag_request_total",
    "Total number of RAG requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# === RAG Pipeline Metrics ===
RETRIEVAL_COUNT = Counter(
    "rag_retrieval_total",
    "Total number of retrieval operations",
    ["retriever_type"],
)

RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval duration in seconds",
    ["retriever_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RETRIEVAL_DOCS = Histogram(
    "rag_retrieval_documents",
    "Number of documents retrieved per query",
    buckets=(1, 3, 5, 10, 15, 20),
)

LLM_TOKEN_USAGE = Counter(
    "rag_llm_token_total",
    "Total LLM tokens used",
    ["type"],  # prompt / completion
)

LLM_DURATION = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation duration in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# === Cache Metrics ===
CACHE_HIT = Counter(
    "rag_cache_hit_total",
    "Semantic cache hits",
)

CACHE_MISS = Counter(
    "rag_cache_miss_total",
    "Semantic cache misses",
)

CACHE_SIZE = Gauge(
    "rag_cache_size",
    "Current semantic cache size",
)

# === Document Processing Metrics ===
DOC_PROCESS_DURATION = Histogram(
    "rag_document_processing_seconds",
    "Document processing duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

DOC_CHUNK_COUNT = Histogram(
    "rag_document_chunk_count",
    "Number of chunks per document",
    buckets=(5, 10, 20, 50, 100, 200),
)

EMBEDDING_BATCH_SIZE = Histogram(
    "rag_embedding_batch_size",
    "Embedding batch size",
    buckets=(1, 10, 32, 64, 128, 256),
)

# === System Metrics ===
ACTIVE_WEBSOCKETS = Gauge(
    "rag_active_websocket_connections",
    "Number of active WebSocket connections",
)

VECTOR_STORE_SIZE = Gauge(
    "rag_vector_store_size",
    "Total vectors in FAISS index",
)

# === Reranker Metrics ===
RERANK_DURATION = Histogram(
    "rag_rerank_duration_seconds",
    "Reranker duration in seconds",
    ["reranker_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RERANK_INPUT_DOCS = Histogram(
    "rag_rerank_input_documents",
    "Number of documents sent to reranker",
    buckets=(1, 3, 5, 10, 15, 20, 50),
)

RERANK_OUTPUT_DOCS = Histogram(
    "rag_rerank_output_documents",
    "Number of documents after reranking",
    buckets=(1, 3, 5, 10, 15, 20),
)

# === Evaluation Metrics ===
EVAL_SCORE = Gauge(
    "rag_evaluation_score",
    "Latest evaluation score",
    ["metric"],
)


def get_metrics() -> bytes:
    """生成 Prometheus 格式的指标输出"""
    return generate_latest()


def get_metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
