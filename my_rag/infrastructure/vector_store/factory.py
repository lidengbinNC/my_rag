"""
向量存储工厂

面试考点：
- 工厂模式：通过配置项 VECTOR_STORE_PROVIDER 切换底层实现
- 开闭原则：新增向量数据库（如 Qdrant、Weaviate）只需新增实现类，不修改调用方
- 延迟导入：只有实际使用的实现才会被 import，避免未安装的依赖报错
"""

from my_rag.infrastructure.vector_store.base import BaseVectorStore


class VectorStoreFactory:

    @staticmethod
    def create(provider: str, dimension: int, **kwargs) -> BaseVectorStore:
        """
        创建向量存储实例

        Args:
            provider: "faiss" 或 "milvus"
            dimension: 向量维度（必须与 Embedding 模型一致）
            **kwargs: 各实现的专属参数
        """
        if provider == "milvus":
            from my_rag.infrastructure.vector_store.milvus_store import MilvusVectorStore
            return MilvusVectorStore(dimension=dimension, **kwargs)

        if provider == "faiss":
            from my_rag.infrastructure.vector_store.faiss_store import FAISSVectorStore
            return FAISSVectorStore(dimension=dimension, **kwargs)

        raise ValueError(
            f"Unknown vector store provider: '{provider}'. "
            f"Supported: 'faiss', 'milvus'"
        )
