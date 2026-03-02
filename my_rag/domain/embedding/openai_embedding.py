"""
OpenAI 兼容 Embedding（支持 OpenAI / DeepSeek / 任何兼容 API）

面试考点：
- OpenAI Embedding API 的调用方式
- 异步 HTTP 客户端 (httpx) 与 API 集成
- 错误重试 + 指数退避策略
- 兼容所有 OpenAI 接口格式的服务（如 vLLM、Ollama）
"""

import asyncio

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(BaseEmbedding):

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dim = dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    response = await client.embeddings.create(model=self._model, input=batch)
                    all_embeddings.extend([d.embedding for d in response.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error("embedding_api_failed", error=str(e))
                        raise
                    wait = 2 ** attempt
                    logger.warning("embedding_api_retry", attempt=attempt, wait=wait, error=str(e))
                    await asyncio.sleep(wait)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed_documents([text])
        return result[0]
