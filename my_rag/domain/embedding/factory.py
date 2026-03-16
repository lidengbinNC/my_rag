"""
Embedding 工厂：根据配置选择本地模型或 API 服务

面试考点：
- 工厂模式：调用方只需传 provider 字符串，无需关心具体实现类
- 延迟导入：只有实际使用的实现才会被 import，避免未安装的依赖报错
- BGE-M3 默认路径：local provider + BAAI/bge-m3 → LocalEmbedding（自动检测并使用 FlagEmbedding）
"""

from my_rag.domain.embedding.base import BaseEmbedding


class EmbeddingFactory:

    @staticmethod
    def create(
        provider: str = "local",
        model: str = "BAAI/bge-m3",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        dimension: int = 1024,
        batch_size: int = 12,
    ) -> BaseEmbedding:
        """
        创建 Embedding 实例

        Args:
            provider: "local"（本地模型）| "openai"（OpenAI 兼容 API）
            model:    模型名称。本地推荐 "BAAI/bge-m3"；OpenAI 推荐 "text-embedding-3-small"
            api_key:  OpenAI API Key（provider="openai" 时必填）
            base_url: API 基础 URL（provider="openai" 时使用）
            dimension: 向量维度（BGE-M3 固定 1024，OpenAI text-embedding-3-small 为 1536）
            batch_size: 批处理大小（BGE-M3 CPU 推荐 8~16，GPU 推荐 32~64）
        """
        if provider == "openai":
            from my_rag.domain.embedding.openai_embedding import OpenAIEmbedding
            return OpenAIEmbedding(model=model, api_key=api_key, base_url=base_url, dim=dimension)

        from my_rag.domain.embedding.local_embedding import LocalEmbedding
        return LocalEmbedding(model_name=model, batch_size=batch_size)
