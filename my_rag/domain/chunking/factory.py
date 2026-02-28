"""分块策略工厂"""

from my_rag.domain.chunking.base import BaseChunker
from my_rag.domain.chunking.fixed_chunker import FixedChunker
from my_rag.domain.chunking.recursive_chunker import RecursiveChunker


_STRATEGY_MAP: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
}


class ChunkerFactory:

    @staticmethod
    def create(
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> BaseChunker:
        chunker_cls = _STRATEGY_MAP.get(strategy)
        if chunker_cls is None:
            supported = ", ".join(sorted(_STRATEGY_MAP.keys()))
            raise ValueError(f"未知分块策略: '{strategy}'，支持: {supported}")
        return chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    @staticmethod
    def available_strategies() -> list[str]:
        return sorted(_STRATEGY_MAP.keys())
