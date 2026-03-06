"""
Reranker 工厂

面试考点：
- 工厂模式：根据配置动态创建不同的 Reranker 实现
- 延迟导入：避免未使用的模型被加载（如不用 Cross-Encoder 就不导入 sentence-transformers）
"""

from my_rag.domain.reranker.base import BaseReranker


class RerankerFactory:

    @staticmethod
    def create(
        provider: str = "cross_encoder",
        model: str = "BAAI/bge-reranker-v2-m3",
        score_threshold: float = 0.0,
        llm=None,
    ) -> BaseReranker:
        """
        创建 Reranker 实例

        Args:
            provider: "cross_encoder" 或 "llm"
            model: Cross-Encoder 模型名称
            score_threshold: 过滤阈值
            llm: LLM 实例（provider="llm" 时必传）
        """
        if provider == "llm":
            if llm is None:
                raise ValueError("LLM Reranker requires an LLM instance")
            from my_rag.domain.reranker.llm_reranker import LLMReranker
            return LLMReranker(
                llm=llm,
                score_threshold=score_threshold,
            )

        from my_rag.domain.reranker.cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name=model,
            score_threshold=score_threshold,
        )
