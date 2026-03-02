"""
本地 Embedding 模型（sentence-transformers）

面试考点：
- sentence-transformers 是 Hugging Face 的语义向量库，支持数百种预训练模型
- 模型加载后常驻内存，避免重复加载的开销（单例模式思想）
- 批处理时需控制 batch_size 避免 GPU/CPU OOM
- normalize_embeddings=True 将向量归一化，使 cosine similarity 等价于 dot product
"""

import asyncio
from functools import lru_cache

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    logger.info("loading_embedding_model", model=model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    logger.info("embedding_model_loaded", model=model_name, dimension=model.get_sentence_embedding_dimension())
    return model


class LocalEmbedding(BaseEmbedding):
    """本地 sentence-transformers Embedding"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed_documents([text])
        return result[0]
