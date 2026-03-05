"""
OpenAI 兼容 LLM 实现

面试考点：
- OpenAI Chat Completions API 的请求/响应格式
- stream=True 返回 SSE 流，每个 chunk 包含 delta.content
- 兼容 OpenAI 接口格式的服务：Ollama、vLLM、DeepSeek、Qwen API 等
- 错误处理：API 限流 (429)、超时、上下文长度超限
"""

from collections.abc import AsyncIterator

from my_rag.domain.llm.base import BaseLLM
from my_rag.utils.logger import get_logger
from my_rag.utils.metrics import LLM_TOKEN_USAGE

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            stream=False,
        )
        if response.usage:
            LLM_TOKEN_USAGE.labels(type="prompt").inc(response.usage.prompt_tokens)
            LLM_TOKEN_USAGE.labels(type="completion").inc(response.usage.completion_tokens)
        return response.choices[0].message.content or ""

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                LLM_TOKEN_USAGE.labels(type="prompt").inc(chunk.usage.prompt_tokens)
                LLM_TOKEN_USAGE.labels(type="completion").inc(chunk.usage.completion_tokens)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
