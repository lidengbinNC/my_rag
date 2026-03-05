"""
Self-RAG（自反思 RAG）—— 面试知识点参考

注意：Self-RAG 的实际实现已集成到 rag_pipeline.py 中，通过 enable_self_rag=True 开启。
本文件仅作为独立版本参考和面试知识点文档。

面试考点（区分度极高）：
- 传统 RAG 的问题：无论什么问题都去检索，有时检索反而引入噪声
- Self-RAG 三次 LLM 判断：
  1. 检索判断（Retrieve Token）：这个问题需要检索吗？
     - "今天几号？" → 不需要检索，LLM 直接回答
     - "公司的退款政策是什么？" → 需要检索
  2. 相关性判断（ISREL Token）：检索到的每篇文档与问题是否相关？过滤噪声
  3. 支撑性判断（ISSUP Token）：生成的回答是否被文档支撑？检测幻觉
     - 如果不通过 → 重新生成或告知用户

- 论文：Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)
- 与 Corrective RAG 的区别：
  - Self-RAG：在生成端做反思（生成后自我评估）
  - Corrective RAG：在检索端做纠正（对检索结果打分、补充）
"""

from dataclasses import dataclass

from my_rag.domain.llm.base import BaseLLM
from my_rag.domain.prompt.template import build_context, build_prompt
from my_rag.domain.retrieval.base import BaseRetriever, RetrievalResult
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

NEED_RETRIEVAL_PROMPT = """判断以下问题是否需要从文档库中检索信息才能回答。

如果是常识性问题、闲聊、或不需要特定文档支持的问题，回答"否"。
如果需要查阅特定文档、数据或专业知识才能准确回答，回答"是"。

问题：{query}

请只回答"是"或"否"，不要解释："""

RELEVANCE_JUDGE_PROMPT = """判断以下文档片段是否与用户问题相关。

用户问题：{query}

文档片段：
{document}

请只回答"相关"或"不相关"："""

SUPPORT_JUDGE_PROMPT = """判断以下回答是否完全由参考文档支撑，没有编造信息。

参考文档：
{context}

回答：
{answer}

请只回答"支撑"或"不支撑"，不要解释："""


@dataclass
class SelfRAGResult:
    answer: str
    sources: list[RetrievalResult]
    needed_retrieval: bool
    relevant_sources: list[RetrievalResult]
    is_supported: bool
    retried: bool = False


class SelfRAGPipeline:
    """Self-RAG Pipeline：判断是否检索 → 检索 → 过滤相关文档 → 生成 → 自我评估"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        max_retries: int = 1,
    ):
        self._retriever = retriever
        self._llm = llm
        self._max_retries = max_retries

    async def run(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        chat_history: str = "",
    ) -> SelfRAGResult:

        needed = await self._judge_need_retrieval(query)

        if not needed:
            logger.info("self_rag_no_retrieval", query=query[:50])
            answer = await self._llm.generate(
                f"请回答以下问题：\n{query}" if not chat_history else
                f"对话历史：\n{chat_history}\n\n请回答：\n{query}"
            )
            return SelfRAGResult(
                answer=answer, sources=[], needed_retrieval=False,
                relevant_sources=[], is_supported=True,
            )

        sources = await self._retriever.retrieve(query, top_k=top_k, knowledge_base_id=knowledge_base_id)

        relevant_sources = await self._filter_relevant(query, sources)

        logger.info(
            "self_rag_relevance_filter",
            total=len(sources),
            relevant=len(relevant_sources),
        )

        if not relevant_sources:
            return SelfRAGResult(
                answer="根据已有文档，暂时无法找到与您问题相关的内容。请尝试换一种方式提问，或上传更多相关文档。",
                sources=sources, needed_retrieval=True,
                relevant_sources=[], is_supported=True,
            )

        for attempt in range(1 + self._max_retries):
            context = build_context([
                {"content": s.content, "source": s.source} for s in relevant_sources
            ])
            prompt = build_prompt(question=query, context=context, chat_history=chat_history)

            answer = await self._llm.generate(prompt)

            is_supported = await self._judge_support(context, answer)

            logger.info(
                "self_rag_support_check",
                attempt=attempt,
                is_supported=is_supported,
            )

            if is_supported or attempt >= self._max_retries:
                return SelfRAGResult(
                    answer=answer, sources=sources, needed_retrieval=True,
                    relevant_sources=relevant_sources, is_supported=is_supported,
                    retried=attempt > 0,
                )

            logger.warning("self_rag_retry", attempt=attempt, reason="answer_not_supported")

        return SelfRAGResult(
            answer=answer, sources=sources, needed_retrieval=True,
            relevant_sources=relevant_sources, is_supported=False, retried=True,
        )

    async def _judge_need_retrieval(self, query: str) -> bool:
        """判断是否需要检索"""
        try:
            result = await self._llm.generate(NEED_RETRIEVAL_PROMPT.format(query=query), temperature=0.0, max_tokens=10)
            return "是" in result
        except Exception:
            return True

    async def _filter_relevant(self, query: str, sources: list[RetrievalResult]) -> list[RetrievalResult]:
        """过滤不相关的检索结果"""
        relevant = []
        for source in sources:
            try:
                result = await self._llm.generate(
                    RELEVANCE_JUDGE_PROMPT.format(query=query, document=source.content[:500]),
                    temperature=0.0, max_tokens=10,
                )
                if "相关" in result and "不相关" not in result:
                    relevant.append(source)
            except Exception:
                relevant.append(source)
        return relevant

    async def _judge_support(self, context: str, answer: str) -> bool:
        """判断回答是否被文档支撑"""
        try:
            result = await self._llm.generate(
                SUPPORT_JUDGE_PROMPT.format(context=context[:2000], answer=answer),
                temperature=0.0, max_tokens=10,
            )
            return "支撑" in result and "不支撑" not in result
        except Exception:
            return True
