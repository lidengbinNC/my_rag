"""
LLM 辅助生成评估数据集

面试考点：
- 为什么用 LLM 生成评估数据？人工标注成本高，LLM 可快速生成大量 QA 对
- 生成流程：从知识库抽取 Chunk → LLM 基于 Chunk 生成 Question + Ground Truth
- 质量控制：多样性采样、去重、可选人工审核
- 局限性：LLM 生成的 Ground Truth 本身可能有偏差，适合冷启动，后续应人工修正
"""

import json
import random
import re
from dataclasses import dataclass

from sqlalchemy import func, select

from my_rag.domain.llm.base import BaseLLM
from my_rag.evaluation.dataset import EvalDataset
from my_rag.infrastructure.database import Chunk, async_session_factory
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

GENERATE_QA_PROMPT = """你是一个专业的评估数据集生成助手。请根据以下文档内容，生成 {num_questions} 个高质量的问答对。

要求：
1. 问题应该多样化：包含事实型、理解型、比较型、应用型等不同类型
2. 问题应该是用户真实可能提出的自然语言问题
3. 答案必须完全基于文档内容，不要引入文档中没有的信息
4. 答案应该完整、准确、简洁
5. 避免生成过于简单或过于模糊的问题

【文档内容】
{context}

请严格按以下 JSON 数组格式输出（不要输出其他内容）：
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}}
]"""

GENERATE_QA_MULTI_CHUNK_PROMPT = """你是一个专业的评估数据集生成助手。以下是来自同一知识库的多段文档内容，请综合这些内容生成 {num_questions} 个高质量的问答对。

要求：
1. 问题应多样化：包含事实型、理解型、比较型、综合分析型等
2. 部分问题可以跨多段内容提问，考察综合理解能力
3. 问题应该是用户真实可能提出的自然语言问题
4. 答案必须完全基于文档内容，不要引入文档中没有的信息
5. 答案应该完整、准确、简洁

{chunks}

请严格按以下 JSON 数组格式输出（不要输出其他内容）：
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}}
]"""


@dataclass
class GenerationConfig:
    """数据集生成配置"""
    num_questions_per_chunk: int = 2
    max_chunks: int = 50
    sampling_strategy: str = "diverse"
    include_multi_chunk_questions: bool = True
    multi_chunk_group_size: int = 3
    num_multi_chunk_questions: int = 3


class DatasetGenerator:
    """LLM 辅助评估数据集生成器

    流程：
    1. 从数据库读取指定知识库的 Chunk
    2. 按策略采样（随机 / 多样性 / 全量）
    3. 对每个 Chunk 调用 LLM 生成 QA 对
    4. 可选：跨 Chunk 组合生成综合性问题
    5. 去重、组装为 EvalDataset
    """

    def __init__(self, llm: BaseLLM, config: GenerationConfig | None = None):
        self._llm = llm
        self._config = config or GenerationConfig()

    async def generate(
        self,
        knowledge_base_id: str,
        dataset_name: str = "",
    ) -> EvalDataset:
        """为指定知识库生成评估数据集"""
        chunks = await self._load_chunks(knowledge_base_id)
        if not chunks:
            logger.warning("no_chunks_found", knowledge_base_id=knowledge_base_id)
            return EvalDataset(name=dataset_name or f"eval_{knowledge_base_id[:8]}")

        sampled = self._sample_chunks(chunks)
        logger.info(
            "chunks_sampled",
            total=len(chunks),
            sampled=len(sampled),
            strategy=self._config.sampling_strategy,
        )

        dataset = EvalDataset(name=dataset_name or f"eval_{knowledge_base_id[:8]}")

        for chunk in sampled:
            qa_pairs = await self._generate_from_chunk(chunk)
            for qa in qa_pairs:
                dataset.add(
                    question=qa["question"],
                    ground_truth=qa["answer"],
                    knowledge_base_id=knowledge_base_id,
                )

        if self._config.include_multi_chunk_questions and len(sampled) >= self._config.multi_chunk_group_size:
            multi_qa = await self._generate_multi_chunk_questions(sampled)
            for qa in multi_qa:
                dataset.add(
                    question=qa["question"],
                    ground_truth=qa["answer"],
                    knowledge_base_id=knowledge_base_id,
                )

        dataset = self._deduplicate(dataset)

        logger.info(
            "dataset_generated",
            name=dataset.name,
            total_samples=len(dataset.samples),
            knowledge_base_id=knowledge_base_id,
        )
        return dataset

    async def _load_chunks(self, knowledge_base_id: str) -> list[dict]:
        """从数据库加载知识库的所有 Chunk"""
        async with async_session_factory() as session:
            stmt = (
                select(Chunk.id, Chunk.content, Chunk.chunk_index, Chunk.document_id)
                .where(Chunk.knowledge_base_id == knowledge_base_id)
                .order_by(Chunk.chunk_index)
            )
            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "id": row.id,
                "content": row.content,
                "chunk_index": row.chunk_index,
                "document_id": row.document_id,
            }
            for row in rows
        ]

    def _sample_chunks(self, chunks: list[dict]) -> list[dict]:
        """按策略采样 Chunk"""
        max_chunks = self._config.max_chunks

        if len(chunks) <= max_chunks:
            return chunks

        strategy = self._config.sampling_strategy

        if strategy == "random":
            return random.sample(chunks, max_chunks)

        if strategy == "diverse":
            return self._diverse_sample(chunks, max_chunks)

        return chunks[:max_chunks]

    def _diverse_sample(self, chunks: list[dict], n: int) -> list[dict]:
        """多样性采样：按文档分组，从每个文档均匀取样"""
        by_doc: dict[str, list[dict]] = {}
        for chunk in chunks:
            doc_id = chunk["document_id"]
            by_doc.setdefault(doc_id, []).append(chunk)

        sampled: list[dict] = []
        doc_ids = list(by_doc.keys())
        per_doc = max(1, n // len(doc_ids))

        for doc_id in doc_ids:
            doc_chunks = by_doc[doc_id]
            if len(doc_chunks) <= per_doc:
                sampled.extend(doc_chunks)
            else:
                step = len(doc_chunks) / per_doc
                sampled.extend(doc_chunks[int(i * step)] for i in range(per_doc))

        if len(sampled) > n:
            sampled = sampled[:n]

        return sampled

    async def _generate_from_chunk(self, chunk: dict) -> list[dict]:
        """对单个 Chunk 调用 LLM 生成 QA 对"""
        prompt = GENERATE_QA_PROMPT.format(
            num_questions=self._config.num_questions_per_chunk,
            context=chunk["content"],
        )

        try:
            response = await self._llm.generate(prompt, temperature=0.7)
            return self._parse_qa_response(response)
        except Exception as e:
            logger.warning("qa_generation_failed", chunk_id=chunk["id"], error=str(e))
            return []

    async def _generate_multi_chunk_questions(self, chunks: list[dict]) -> list[dict]:
        """跨多个 Chunk 生成综合性问题"""
        group_size = self._config.multi_chunk_group_size
        group = random.sample(chunks, min(group_size, len(chunks)))

        chunks_text = "\n\n".join(
            f"【文档片段 {i + 1}】\n{c['content']}"
            for i, c in enumerate(group)
        )

        prompt = GENERATE_QA_MULTI_CHUNK_PROMPT.format(
            num_questions=self._config.num_multi_chunk_questions,
            chunks=chunks_text,
        )

        try:
            response = await self._llm.generate(prompt, temperature=0.7)
            return self._parse_qa_response(response)
        except Exception as e:
            logger.warning("multi_chunk_generation_failed", error=str(e))
            return []

    @staticmethod
    def _parse_qa_response(response: str) -> list[dict]:
        """从 LLM 输出中解析 QA JSON 数组"""
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        qa_pairs = []
        for item in items:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                q = item["question"].strip()
                a = item["answer"].strip()
                if q and a:
                    qa_pairs.append({"question": q, "answer": a})
        return qa_pairs

    @staticmethod
    def _deduplicate(dataset: EvalDataset) -> EvalDataset:
        """基于问题文本去重"""
        seen: set[str] = set()
        unique_samples = []
        for sample in dataset.samples:
            normalized = sample.question.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_samples.append(sample)
        dataset.samples = unique_samples
        return dataset
