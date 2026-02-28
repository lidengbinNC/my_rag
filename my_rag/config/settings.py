"""
配置管理模块 - 基于 Pydantic Settings

面试考点：
- Pydantic Settings 实现多环境配置
- 环境变量优先级：.env 文件 < 系统环境变量 < 代码默认值
- 配置分组与嵌套
- 类型安全的配置管理（对比 Java 的 @ConfigurationProperties）
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用基础配置"""
    name: str = Field(default="MyRAG", alias="APP_NAME")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=True, alias="APP_DEBUG")
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    url: str = Field(
        default="sqlite+aiosqlite:///./data/myrag.db",
        alias="DATABASE_URL",
    )
    echo: bool = Field(default=False, alias="DATABASE_ECHO")


class LLMSettings(BaseSettings):
    """LLM 配置"""
    provider: str = Field(default="openai", alias="LLM_PROVIDER")
    model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    api_key: str = Field(default="", alias="LLM_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")


class EmbeddingSettings(BaseSettings):
    """Embedding 配置"""
    provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")


class ChunkSettings(BaseSettings):
    """分块配置"""
    size: int = Field(default=512, alias="CHUNK_SIZE")
    overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    strategy: str = Field(default="recursive", alias="CHUNK_STRATEGY")


class StorageSettings(BaseSettings):
    """文件存储配置"""
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    max_file_size: int = Field(default=50 * 1024 * 1024, alias="MAX_FILE_SIZE")


class Settings(BaseSettings):
    """全局配置聚合"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
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
