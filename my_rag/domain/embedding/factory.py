"""Embedding 工厂：根据配置选择本地模型或 API 服务"""

from my_rag.domain.embedding.base import BaseEmbedding


class EmbeddingFactory:

    @staticmethod
    def create(
        provider: str = "local",
        model: str = "BAAI/bge-small-zh-v1.5",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        dimension: int = 1024,
    ) -> BaseEmbedding:
        if provider == "openai":
            from my_rag.domain.embedding.openai_embedding import OpenAIEmbedding
            return OpenAIEmbedding(model=model, api_key=api_key, base_url=base_url, dim=dimension)

        from my_rag.domain.embedding.local_embedding import LocalEmbedding
        return LocalEmbedding(model_name=model)
