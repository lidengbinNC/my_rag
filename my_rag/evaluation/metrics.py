"""
RAGAS 风格评估指标

面试考点（高频必考）：
- RAG 评估五大指标的定义、计算方式、适用场景
- 为什么需要自动化评估？手动评估成本高、不可重复、主观性强

1. Faithfulness（忠实度）
   - 回答是否忠实于检索到的文档？有没有"幻觉"？
   - 做法：将回答拆成多个陈述句，逐一判断是否有文档证据支撑
   - 分数 = 有支撑的陈述数 / 总陈述数

2. Answer Relevancy（回答相关性）
   - 回答是否和用户问题相关？有没有答非所问？
   - 做法：从回答反向生成 N 个问题，计算与原问题的平均余弦相似度
   - 分数 = avg(cosine_sim(原问题, 反向生成的问题))

3. Context Precision（上下文精度）
   - 检索到的文档中，相关文档是否排在前面？
   - 分数 = 加权精度（排名靠前的相关文档权重更高）

4. Context Recall（上下文召回）
   - Ground Truth 中的信息有多少被检索到的文档覆盖？
   - 做法：将 Ground Truth 拆成陈述句，判断每句是否能从检索文档中找到
   - 分数 = 被覆盖的陈述数 / 总陈述数

5. Answer Correctness（回答正确性）
   - 回答与标准答案（Ground Truth）的一致程度
   - 分数 = 语义相似度（embedding cosine）和事实重叠的加权
"""

from dataclasses import dataclass, field

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

FAITHFULNESS_PROMPT = """请判断以下【回答】中的每个陈述是否都有【参考文档】的支撑。

【参考文档】
{context}

【回答】
{answer}

请按以下格式输出：
- 总陈述数：N
- 有支撑的陈述数：M
- 分数：M/N = X.XX
- 无支撑的陈述（如有）：列出具体内容

请只输出 JSON 格式：
{{"total_statements": N, "supported_statements": M, "score": X.XX, "unsupported": ["..."]}}"""

RELEVANCY_PROMPT = """根据以下回答，生成 3 个可能产生这个回答的问题。

回答：{answer}

请每行输出一个问题，不要编号："""

RECALL_PROMPT = """请判断以下【标准答案】中的每个信息点是否能从【检索文档】中找到。

【检索文档】
{context}

【标准答案】
{ground_truth}

请只输出 JSON 格式：
{{"total_points": N, "covered_points": M, "score": X.XX}}"""


@dataclass
class EvaluationResult:
    """单条评估结果"""
    question: str
    answer: str
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0

    @property
    def overall_score(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision, self.context_recall]
        valid = [s for s in scores if s > 0]
        return sum(valid) / len(valid) if valid else 0.0


@dataclass
class EvaluationReport:
    """评估报告"""
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        vals = [r.faithfulness for r in self.results if r.faithfulness > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_relevancy(self) -> float:
        vals = [r.answer_relevancy for r in self.results if r.answer_relevancy > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_precision(self) -> float:
        vals = [r.context_precision for r in self.results if r.context_precision > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_recall(self) -> float:
        vals = [r.context_recall for r in self.results if r.context_recall > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_correctness(self) -> float:
        vals = [r.answer_correctness for r in self.results if r.answer_correctness > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {
            "total_samples": len(self.results),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevancy": round(self.avg_relevancy, 4),
            "avg_context_precision": round(self.avg_precision, 4),
            "avg_context_recall": round(self.avg_recall, 4),
            "avg_answer_correctness": round(self.avg_correctness, 4),
        }


class RAGEvaluator:
    """RAGAS 风格 RAG 评估器"""

    def __init__(self, llm: BaseLLM, embedding: BaseEmbedding):
        self._llm = llm
        self._embedding = embedding

    async def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """忠实度：回答是否有文档支撑"""
        context_text = "\n\n".join(contexts)
        prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
        try:
            result = await self._llm.generate(prompt, temperature=0.0)
            score = self._extract_score(result)
            return score
        except Exception as e:
            logger.warning("faithfulness_eval_failed", error=str(e))
            return 0.0

    async def evaluate_relevancy(self, question: str, answer: str) -> float:
        """回答相关性：从回答反向生成问题，与原问题比较"""
        prompt = RELEVANCY_PROMPT.format(answer=answer)
        try:
            result = await self._llm.generate(prompt, temperature=0.3)
            generated_questions = [q.strip() for q in result.strip().split("\n") if q.strip()][:3]

            if not generated_questions:
                return 0.0

            original_emb = await self._embedding.embed_query(question)
            gen_embs = await self._embedding.embed_documents(generated_questions)

            import numpy as np
            original_vec = np.array(original_emb)
            similarities = [
                float(np.dot(original_vec, np.array(gen_emb)) / (np.linalg.norm(original_vec) * np.linalg.norm(gen_emb) + 1e-8))
                for gen_emb in gen_embs
            ]
            return max(0.0, min(1.0, sum(similarities) / len(similarities)))
        except Exception as e:
            logger.warning("relevancy_eval_failed", error=str(e))
            return 0.0

    async def evaluate_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """上下文召回：标准答案中多少信息被检索文档覆盖"""
        if not ground_truth:
            return 0.0
        context_text = "\n\n".join(contexts)
        prompt = RECALL_PROMPT.format(context=context_text, ground_truth=ground_truth)
        try:
            result = await self._llm.generate(prompt, temperature=0.0)
            score = self._extract_score(result)
            return score
        except Exception as e:
            logger.warning("recall_eval_failed", error=str(e))
            return 0.0

    async def evaluate_correctness(self, answer: str, ground_truth: str) -> float:
        """回答正确性：与标准答案的语义相似度"""
        if not ground_truth:
            return 0.0
        try:
            embs = await self._embedding.embed_documents([answer, ground_truth])
            import numpy as np
            a, b = np.array(embs[0]), np.array(embs[1])
            similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning("correctness_eval_failed", error=str(e))
            return 0.0

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvaluationResult:
        """运行全部评估指标"""
        result = EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
        )

        result.faithfulness = await self.evaluate_faithfulness(answer, contexts)
        result.answer_relevancy = await self.evaluate_relevancy(question, answer)

        if ground_truth:
            result.context_recall = await self.evaluate_context_recall(ground_truth, contexts)
            result.answer_correctness = await self.evaluate_correctness(answer, ground_truth)

        logger.info(
            "evaluation_completed",
            question=question[:50],
            faithfulness=result.faithfulness,
            relevancy=result.answer_relevancy,
            recall=result.context_recall,
            correctness=result.answer_correctness,
        )

        return result

    @staticmethod
    def _normalize_score(raw: float) -> float:
        """将分数归一化到 [0, 1] 区间

        LLM 有时返回百分制 (e.g. 85.0) 而非小数 (e.g. 0.85)，
        这里统一处理：> 1.0 的值视为百分制，除以 100。
        """
        if raw > 1.0:
            raw = raw / 100.0
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _extract_score(llm_output: str) -> float:
        """从 LLM 输出中提取 score 字段"""
        import json
        import re
        try:
            json_match = re.search(r'\{[^}]+\}', llm_output)
            if json_match:
                data = json.loads(json_match.group())
                raw = float(data.get("score", 0.0))
                return RAGEvaluator._normalize_score(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        score_match = re.search(r'score["\s:]+(\d+\.?\d*)', llm_output, re.IGNORECASE)
        if score_match:
            raw = float(score_match.group(1))
            return RAGEvaluator._normalize_score(raw)
        return 0.0
