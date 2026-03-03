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

    url: str = "sqlite+aiosqlite:///./data/myrag.db"
    echo: bool = False


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
    model: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8",
        env_prefix="RETRIEVAL_", extra="ignore",
    )

    top_k: int = 5
    rrf_k: int = 60


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
