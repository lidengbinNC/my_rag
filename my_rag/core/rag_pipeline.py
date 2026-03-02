"""
RAG Pipeline 编排器

面试考点：
- Pipeline 模式：将 Query → Retrieve → (Rerank) → Generate 组合为可配置的流水线
- 各环节职责清晰、可独立替换（依赖倒置原则）
- 流式输出：AsyncIterator 逐 token 传递给前端
- 上下文窗口管理：检索结果 + 历史对话 + Query 需控制在 LLM 的 max context 内
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.prompt.template import build_context, build_prompt
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]
    prompt: str = ""
    usage: dict = field(default_factory=dict)


class RAGPipeline:
    """RAG Pipeline：Retrieve → Build Prompt → Generate"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
    ):
        self._retriever = retriever
        self._llm = llm

    async def run(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> RAGResponse:
        """同步执行完整 RAG 流程"""
        sources = await self._retriever.retrieve(
            query, top_k=top_k, knowledge_base_id=knowledge_base_id
        )

        logger.info("rag_retrieval_done", query=query[:50], source_count=len(sources))

        context = build_context([
            {"content": s.content, "source": s.source} for s in sources
        ])
        prompt = build_prompt(question=query, context=context, chat_history=chat_history)

        answer = await self._llm.generate(prompt)

        logger.info("rag_generation_done", answer_length=len(answer))

        return RAGResponse(answer=answer, sources=sources, prompt=prompt)

    async def stream(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> AsyncIterator[dict]:
        """流式 RAG：先返回检索结果，再逐 token 返回生成内容"""

        sources = await self._retriever.retrieve(
            query, top_k=top_k, knowledge_base_id=knowledge_base_id
        )

        yield {
            "type": "retrieval",
            "documents": [
                {"content": s.content, "source": s.source, "score": s.score, "chunk_id": s.chunk_id}
                for s in sources
            ],
        }

        context = build_context([
            {"content": s.content, "source": s.source} for s in sources
        ])
        prompt = build_prompt(question=query, context=context, chat_history=chat_history)

        full_answer = ""
        async for token in self._llm.stream_generate(prompt):
            full_answer += token
            yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "full_answer": full_answer,
            "sources": [
                {"content": s.content, "source": s.source, "score": s.score, "chunk_id": s.chunk_id}
                for s in sources
            ],
        }
