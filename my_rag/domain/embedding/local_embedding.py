"""
本地 Embedding 模型（sentence-transformers）

面试考点：
- sentence-transformers 是 Hugging Face 的语义向量库，支持数百种预训练模型
- 模型加载后常驻内存，避免重复加载的开销（单例模式思想）
- 批处理时需控制 batch_size 避免 GPU/CPU OOM
- normalize_embeddings=True 将向量归一化，使 cosine similarity 等价于 dot product

BGE-M3 特性（面试考点）：
- 支持三种向量类型：Dense（稠密）/ Sparse（稀疏 SPLADE 风格）/ ColBERT（多向量）
- Dense 向量维度：1024，用于语义相似度检索
- Sparse 向量：词粒度权重，类似 BM25，用于关键词精确匹配
- 混合检索时 Dense + Sparse 互补，可显著提升召回率
- embed_query 与 embed_documents 使用相同编码方式（BGE-M3 不区分 instruction）
- 需要通过 FlagEmbedding 库加载才能同时获取 Dense + Sparse 输出
"""

import asyncio
from functools import lru_cache

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

# BGE-M3 模型标识（用于识别是否启用多向量模式）
_BGE_M3_MODEL_ID = "BAAI/bge-m3"


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    """加载 sentence-transformers 模型（通用，支持所有 ST 兼容模型）"""
    from sentence_transformers import SentenceTransformer
    logger.info("loading_embedding_model", model=model_name, backend="sentence_transformers")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    logger.info(
        "embedding_model_loaded",
        model=model_name,
        dimension=model.get_sentence_embedding_dimension(),
    )
    return model


@lru_cache(maxsize=2)
def _load_flag_model(model_name: str):
    """
    加载 FlagEmbedding BGEM3FlagModel（专为 BGE-M3 设计）

    FlagEmbedding 相比 sentence-transformers 的优势：
    - 同时输出 Dense + Sparse + ColBERT 三种向量
    - Sparse 向量用于混合检索，弥补纯语义检索对关键词匹配的不足
    - use_fp16=True 使用半精度推理，速度提升约 2x，精度损失可忽略
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("loading_bge_m3_model", model=model_name, backend="FlagEmbedding")
        model = BGEM3FlagModel(model_name, use_fp16=True)
        logger.info("bge_m3_model_loaded", model=model_name)
        return model
    except ImportError:
        logger.warning(
            "FlagEmbedding_not_installed",
            hint="pip install FlagEmbedding; falling back to sentence-transformers (dense only)",
        )
        return None


class LocalEmbedding(BaseEmbedding):
    """
    本地 Embedding 模型，支持两种后端：

    1. sentence-transformers（通用）：适合 bge-small/large 等模型
    2. FlagEmbedding（BGE-M3 专用）：同时输出 Dense + Sparse 向量

    自动检测：model_name 包含 "bge-m3" 时优先使用 FlagEmbedding
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 12,
        use_sparse: bool = False,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        # use_sparse=True 时同时计算稀疏向量（仅 BGE-M3 支持）
        self._use_sparse = use_sparse and ("bge-m3" in model_name.lower())
        self._st_model = None
        self._flag_model = None

    def _is_bge_m3(self) -> bool:
        return "bge-m3" in self._model_name.lower()

    @property
    def supports_sparse(self) -> bool:
        return self._is_bge_m3()

    def _get_flag_model(self):
        """获取 FlagEmbedding 模型（BGE-M3 专用）"""
        if self._flag_model is None:
            self._flag_model = _load_flag_model(self._model_name)
        return self._flag_model

    def _get_st_model(self):
        """获取 sentence-transformers 模型（通用回退）"""
        if self._st_model is None:
            self._st_model = _load_sentence_transformer(self._model_name)
        return self._st_model

    @property
    def dimension(self) -> int:
        if self._is_bge_m3():
            flag_model = self._get_flag_model()
            if flag_model is not None:
                # BGE-M3 Dense 向量固定 1024 维
                return 1024
        return self._get_st_model().get_sentence_embedding_dimension()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量文档向量化（返回 Dense 向量）

        BGE-M3 路径：使用 FlagEmbedding 编码，batch_size 建议 8~16（显存限制）
        通用路径：使用 sentence-transformers 编码
        """
        if self._is_bge_m3():
            flag_model = self._get_flag_model()
            if flag_model is not None:
                return await self._flag_encode(flag_model, texts)

        model = self._get_st_model()
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """
        查询向量化

        BGE-M3 不区分 query/document instruction（与 bge-large-zh 不同）
        直接复用 embed_documents 即可
        """
        result = await self.embed_documents([text])
        return result[0]

    async def embed_documents_with_sparse(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        BGE-M3 专用：同时返回 Dense 向量 + Sparse 向量

        面试考点：
        - Dense 向量：1024 维 float，用于 ANN 语义检索
        - Sparse 向量：{token_id: weight} 字典，类似 SPLADE，用于关键词检索
        - Milvus 2.4+ 原生支持混合检索（dense + sparse），无需额外 BM25 索引

        Returns:
            (dense_embeddings, sparse_embeddings)
            sparse_embeddings: List of {token_id: weight} dicts
        """
        if not self._is_bge_m3():
            raise NotImplementedError("Sparse embeddings are only supported by BGE-M3")

        flag_model = self._get_flag_model()
        if flag_model is None:
            raise RuntimeError(
                "FlagEmbedding is required for sparse embeddings. "
                "Install with: pip install FlagEmbedding"
            )

        def _encode():
            return flag_model.encode(
                texts,
                batch_size=self._batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        output = await asyncio.to_thread(_encode)
        dense = [v.tolist() for v in output["dense_vecs"]]
        sparse = [dict(zip(sv.indices.tolist(), sv.values.tolist()))
                  for sv in output["lexical_weights"]]
        return dense, sparse

    async def embed_query_with_sparse(
        self, text: str
    ) -> tuple[list[float], dict[int, float]]:
        """BGE-M3 专用：查询的 Dense + Sparse 向量"""
        dense_list, sparse_list = await self.embed_documents_with_sparse([text])
        return dense_list[0], sparse_list[0]

    async def embed_documents_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        return await self.embed_documents_with_sparse(texts)

    async def embed_query_hybrid(self, text: str) -> tuple[list[float], dict[int, float]]:
        return await self.embed_query_with_sparse(text)

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    async def _flag_encode(self, flag_model, texts: list[str]) -> list[list[float]]:
        """使用 FlagEmbedding 编码，只返回 Dense 向量"""
        def _encode():
            return flag_model.encode(
                texts,
                batch_size=self._batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

        output = await asyncio.to_thread(_encode)
        return [v.tolist() for v in output["dense_vecs"]]
