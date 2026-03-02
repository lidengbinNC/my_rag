"""
LLM 抽象基类

面试考点：
- 同步生成 vs 流式生成（SSE / WebSocket）
- AsyncIterator 实现 token-by-token 流式输出
- temperature / max_tokens / top_p 等采样参数的含义
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """同步生成完整回答"""
        ...

    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成，逐 token 返回"""
        ...
