"""LLM 工厂"""

from my_rag.domain.llm.base import BaseLLM


class LLMFactory:

    @staticmethod
    def create(
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> BaseLLM:
        from my_rag.domain.llm.openai_llm import OpenAILLM
        return OpenAILLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
