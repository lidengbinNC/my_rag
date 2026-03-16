"""
配置管理模块 - 基于 Pydantic Settings

面试考点：
- Pydantic Settings 的 env_prefix 实现分组环境变量读取
- env_file 使用绝对路径避免 CWD 不一致的问题
- 嵌套 BaseSettings 子类需各自声明 env_file，父类不会传递
- 环境变量优先级：系统环境变量 > .env 文件 > 代码默认值
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="APP_", extra="ignore",
    )

    name: str = "MyRAG"
    version: str = "0.1.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="DATABASE_", extra="ignore",
    )

    # MySQL 异步连接串格式：mysql+aiomysql://user:password@host:3306/dbname?charset=utf8mb4
    url: str = "mysql+aiomysql://myrag:myrag123@localhost:3306/myrag?charset=utf8mb4"
    echo: bool = False

    # 连接池参数（面试考点：连接池调优）
    pool_size: int = 10          # 常驻连接数，建议 = CPU 核心数 * 2
    max_overflow: int = 20       # 超出 pool_size 后允许额外创建的连接数
    pool_timeout: int = 30       # 等待连接超时秒数
    pool_recycle: int = 3600     # 连接最大存活秒数（小于 MySQL wait_timeout=28800）


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="LLM_", extra="ignore",
    )

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 2048
    temperature: float = 0.7


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="EMBEDDING_", extra="ignore",
    )

    provider: str = "local"
    # BGE-M3：多功能多语言模型，Dense 维度 1024，支持 Dense + Sparse 混合检索
    model: str = "BAAI/bge-m3"
    dimension: int = 1024


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="RETRIEVAL_", extra="ignore",
    )

    top_k: int = 5
    rrf_k: int = 60
    enable_hyde: bool = False
    enable_multi_query: bool = False
    enable_cache: bool = True
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600
    enable_rerank: bool = False
    rerank_provider: str = "cross_encoder"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.0


class ChunkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="CHUNK_", extra="ignore",
    )

    size: int = 512
    overlap: int = 50
    strategy: str = "recursive"


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        extra="ignore",
    )

    upload_dir: str = "./data/uploads"
    max_file_size: int = 50 * 1024 * 1024


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="VECTOR_STORE_", extra="ignore",
    )

    # "faiss" 或 "milvus"
    provider: str = "faiss"

    # ── Milvus 连接 ──
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_db_name: str = "default"
    # 企业版 / Zilliz Cloud 使用 URI + Token 方式
    milvus_uri: str = ""
    milvus_token: str = ""

    # ── Collection 设计 ──
    milvus_collection: str = "rag_chunks"
    # IVF_FLAT / HNSW / IVF_SQ8（生产推荐 HNSW）
    milvus_index_type: str = "HNSW"
    # IP（内积，适合归一化向量） / L2
    milvus_metric_type: str = "IP"
    # HNSW 参数：M 越大召回越准但内存越多，efConstruction 越大建索引越慢但质量越好
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 256
    # 搜索时 ef 越大召回越准但越慢（通常 ef >= top_k * 2）
    milvus_hnsw_ef_search: int = 64


class DingTalkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="DINGTALK_", extra="ignore",
    )

    enabled: bool = False
    webhook_url: str = ""
    secret: str = ""


class Settings(BaseSettings):
    """全局配置聚合"""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    chunk: ChunkSettings = ChunkSettings()
    storage: StorageSettings = StorageSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    dingtalk: DingTalkSettings = DingTalkSettings()

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_dir(self) -> Path:
        d = self.base_dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def upload_path(self) -> Path:
        p = Path(self.storage.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
