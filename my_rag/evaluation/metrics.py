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

6. Exact Match（精确匹配）—— HotpotQA 专用
   - 规范化后完全匹配，是 HotpotQA 官方主指标
   - 规范化：小写 + 去冠词/标点/多余空格

7. Token F1（词级 F1）—— HotpotQA 专用
   - 分词后计算预测词与标准答案词的 F1
   - 比 EM 宽松，能捕获部分正确的情况
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from my_rag.domain.embedding.base import BaseEmbedding
from my_rag.domain.llm.base import BaseLLM
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。请判断【回答】中每个陈述是否有【参考文档】的明确支撑。

【参考文档】
{context}

【回答】
{answer}

评估规则（严格执行）：
- 将回答拆分为独立的陈述句
- 每个陈述必须能在参考文档中找到明确依据才算"有支撑"
- 推断、延伸或文档未提及的内容均视为"无支撑"
- 若回答为空或无实质内容，total_statements=0，score=0.0

请只输出如下 JSON，不要有任何其他文字：
{{"total_statements": N, "supported_statements": M, "score": M_divided_by_N, "unsupported": ["无支撑的陈述1", "..."]}}

注意：score 必须是 0 到 1 之间的小数（如 0.75），不是百分数。"""

RELEVANCY_PROMPT = """根据以下【回答】，生成 3 个最可能触发该回答的用户问题。

【回答】
{answer}

要求：
- 每行输出一个问题
- 不要编号，不要解释
- 问题应自然、简洁，像真实用户提问"""

RECALL_PROMPT = """你是一个严格的信息核查员。请判断【标准答案】中的每个信息点是否能从【检索文档】中找到。

【检索文档】
{context}

【标准答案】
{ground_truth}

评估规则（严格执行）：
- 将标准答案拆分为独立的信息点
- 每个信息点必须在检索文档中有明确对应内容才算"已覆盖"
- 语义相近但细节不同的内容不算覆盖
- 若标准答案为空，total_points=0，score=0.0

请只输出如下 JSON，不要有任何其他文字：
{{"total_points": N, "covered_points": M, "score": M_divided_by_N}}

注意：score 必须是 0 到 1 之间的小数（如 0.75），不是百分数。"""

CORRECTNESS_PROMPT = """你是一个严格的事实核查员。请对比【回答】与【标准答案】的事实重叠程度。

【标准答案】
{ground_truth}

【回答】
{answer}

评估规则（严格执行）：
- 从标准答案中提取所有关键事实点（人名、地名、数字、时间、结论等）
- 判断每个事实点是否在回答中被正确提及（必须事实一致，不能只是语义相近）
- 回答中有额外错误信息不影响已正确覆盖的得分，但不加分
- 若标准答案极短（如单个词/短语），直接判断回答是否包含该内容

请只输出如下 JSON，不要有任何其他文字：
{{"total_facts": N, "correct_facts": M, "score": M_divided_by_N}}

注意：score 必须是 0 到 1 之间的小数（如 0.75），不是百分数。"""


def normalize_answer(text: str) -> str:
    """HotpotQA 官方规范化：小写 + 去冠词 + 去标点 + 合并空格"""
    import re
    import string
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """HotpotQA 官方 Exact Match：规范化后完全匹配返回 1.0，否则 0.0"""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """HotpotQA 官方 Token F1：词级别 precision/recall/F1"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvaluationResult:
    """单条评估结果

    各指标使用 Optional[float]：
    - None  表示该指标未被计算（如无 ground_truth 时不计算 recall/correctness）
    - 0.0   表示计算结果确实为零（如回答完全不相关）
    两者语义不同，不可混用，避免零值过滤导致平均分虚高。

    HotpotQA 专用字段（exact_match / token_f1）：
    - 不依赖 LLM，纯字符串匹配，结果客观可重复
    - 是 HotpotQA 官方评估主指标，优先参考
    """
    question: str
    answer: str
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    # HotpotQA 官方指标（有 ground_truth 时自动计算，不依赖 LLM）
    exact_match: Optional[float] = None
    token_f1: Optional[float] = None

    @property
    def overall_score(self) -> float:
        """所有已计算指标（非 None）的简单平均，含 answer_correctness"""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
            self.answer_correctness,
        ]
        valid = [s for s in scores if s is not None]
        return sum(valid) / len(valid) if valid else 0.0


@dataclass
class EvaluationReport:
    """评估报告"""
    results: list[EvaluationResult] = field(default_factory=list)

    @staticmethod
    def _avg(vals: list[Optional[float]]) -> float:
        """对已计算的指标（非 None）求平均；真实零分参与计算，未计算的 None 不参与"""
        computed = [v for v in vals if v is not None]
        return sum(computed) / len(computed) if computed else 0.0

    @property
    def avg_faithfulness(self) -> float:
        return self._avg([r.faithfulness for r in self.results])

    @property
    def avg_relevancy(self) -> float:
        return self._avg([r.answer_relevancy for r in self.results])

    @property
    def avg_precision(self) -> float:
        return self._avg([r.context_precision for r in self.results])

    @property
    def avg_recall(self) -> float:
        return self._avg([r.context_recall for r in self.results])

    @property
    def avg_correctness(self) -> float:
        return self._avg([r.answer_correctness for r in self.results])

    @property
    def avg_exact_match(self) -> float:
        return self._avg([r.exact_match for r in self.results])

    @property
    def avg_token_f1(self) -> float:
        return self._avg([r.token_f1 for r in self.results])

    def summary(self) -> dict:
        d = {
            "total_samples": len(self.results),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevancy": round(self.avg_relevancy, 4),
            "avg_context_precision": round(self.avg_precision, 4),
            "avg_context_recall": round(self.avg_recall, 4),
            "avg_answer_correctness": round(self.avg_correctness, 4),
        }
        # 仅当有样本计算了 EM/F1 时才加入汇总
        if any(r.exact_match is not None for r in self.results):
            d["avg_exact_match"] = round(self.avg_exact_match, 4)
            d["avg_token_f1"] = round(self.avg_token_f1, 4)
        return d


class RAGEvaluator:
    """RAGAS 风格 RAG 评估器"""

    def __init__(self, llm: BaseLLM, embedding: BaseEmbedding):
        self._llm = llm
        self._embedding = embedding

    async def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> Optional[float]:
        """忠实度：回答是否有文档支撑；LLM 调用失败返回 None（未计算）"""
        if not answer or not contexts:
            return None
        context_text = "\n\n".join(contexts)
        prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
        try:
            result = await self._llm.generate(prompt, temperature=0.0)
            return self._extract_score(result)
        except Exception as e:
            logger.warning("faithfulness_eval_failed", error=str(e))
            return None

    async def evaluate_relevancy(self, question: str, answer: str) -> Optional[float]:
        """回答相关性：从回答反向生成问题，与原问题比较；失败返回 None"""
        prompt = RELEVANCY_PROMPT.format(answer=answer)
        try:
            import numpy as np
            result = await self._llm.generate(prompt, temperature=0.3)
            generated_questions = [q.strip() for q in result.strip().split("\n") if q.strip()][:3]

            if not generated_questions:
                return 0.0

            original_emb = await self._embedding.embed_query(question)
            gen_embs = await self._embedding.embed_documents(generated_questions)

            original_vec = np.array(original_emb)
            similarities = [
                float(np.dot(original_vec, np.array(gen_emb)) / (np.linalg.norm(original_vec) * np.linalg.norm(gen_emb) + 1e-8))
                for gen_emb in gen_embs
            ]
            return max(0.0, min(1.0, sum(similarities) / len(similarities)))
        except Exception as e:
            logger.warning("relevancy_eval_failed", error=str(e))
            return None

    async def evaluate_context_recall(self, ground_truth: str, contexts: list[str], supporting_titles: list[str]) -> Optional[float]:
        """上下文召回：标准答案中多少信息被检索文档覆盖；无 ground_truth 或失败返回 None"""
        if not ground_truth:
            return None

        if not supporting_titles:
            return 0.0

        supporting_set = set(supporting_titles)
        title_set = set()
        for context in contexts:
            parts = context.split("\n\n", 1)
            if parts:
                title_set.add(parts[0])

        hit_count = sum(1 for title in title_set if title in supporting_set)
        result = hit_count / len(supporting_titles) if supporting_titles else 0.0
        return result

    async def evaluate_correctness(self, answer: str, ground_truth: str) -> Optional[float]:
        """回答正确性：事实重叠（LLM 判断，权重 0.75）+ 语义相似度（embedding，权重 0.25）

        仅用语义相似度会导致"语义相近但事实相反"的回答得高分（如"北京"vs"上海"），
        加入 LLM 事实重叠判断后，分数更能反映真实正确性。
        与 RAGAS 原版保持一致：correctness = 0.75 * fact_score + 0.25 * semantic_score
        """
        if not ground_truth:
            return None
        import numpy as np
        try:
            # 语义相似度
            embs = await self._embedding.embed_documents([answer, ground_truth])
            a, b = np.array(embs[0]), np.array(embs[1])
            semantic_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            semantic_score = max(0.0, min(1.0, semantic_score))
        except Exception as e:
            logger.warning("correctness_semantic_failed", error=str(e))
            semantic_score = 0.0

        try:
            # 事实重叠：LLM 判断标准答案中的事实点有多少被回答正确覆盖
            prompt = CORRECTNESS_PROMPT.format(answer=answer, ground_truth=ground_truth)
            llm_result = await self._llm.generate(prompt, temperature=0.0)
            fact_score = self._extract_score(llm_result)
        except Exception as e:
            logger.warning("correctness_fact_failed", error=str(e))
            # LLM 调用失败时退化为纯语义相似度
            fact_score = semantic_score

        score = 0.75 * fact_score + 0.25 * semantic_score
        return max(0.0, min(1.0, score))

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
        supporting_titles: list[str]=None
    ) -> EvaluationResult:
        """并发运行全部评估指标

        4 个指标之间无依赖关系，用 asyncio.gather 同时发出所有 LLM/embedding 请求，
        总耗时 ≈ 最慢单个指标的耗时，而非 4 个指标串行之和。
        """
        import asyncio
        import time

        async def _skip() -> None:
            return None

        t0 = time.perf_counter()

        # 无 ground_truth 时 recall/correctness 直接跳过，不发起 LLM 调用
        faithfulness, relevancy, recall, correctness = await asyncio.gather(
            self.evaluate_faithfulness(answer, contexts),
            self.evaluate_relevancy(question, answer),
            self.evaluate_context_recall(ground_truth, contexts,supporting_titles) if ground_truth else _skip(),
            self.evaluate_correctness(answer, ground_truth) if ground_truth else _skip(),
        )

        # HotpotQA 官方指标：纯字符串匹配，不依赖 LLM，有 ground_truth 时必算
        em = exact_match_score(answer, ground_truth) if ground_truth else None
        f1 = token_f1_score(answer, ground_truth) if ground_truth else None

        elapsed = time.perf_counter() - t0
        logger.info(
            "evaluation_completed",
            question=question[:50],
            faithfulness=faithfulness,
            relevancy=relevancy,
            recall=recall,
            correctness=correctness,
            exact_match=em,
            token_f1=f1,
            elapsed_s=round(elapsed, 2),
        )

        return EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_recall=recall,
            answer_correctness=correctness,
            exact_match=em,
            token_f1=f1,
        )

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
        """从 LLM 输出中提取 score 字段

        使用逐字符括号匹配而非 r'\\{[^}]+\\}' 正则，正确处理含数组的嵌套 JSON
        （如 faithfulness prompt 返回的 "unsupported": ["..."] 结构）。
        降级策略：JSON 解析失败时用正则直接提取 score 数值。
        """
        import json
        import re

        # 找到第一个 '{' 后用括号计数扫描到配对的 '}'，支持嵌套
        start = llm_output.find('{')
        if start != -1:
            depth = 0
            for i, ch in enumerate(llm_output[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(llm_output[start:i + 1])
                            raw = float(data.get("score", 0.0))
                            return RAGEvaluator._normalize_score(raw)
                        except (json.JSONDecodeError, ValueError):
                            break

        # 降级：直接从文本中提取 score 后的数字
        score_match = re.search(r'score["\s:]+(\d+\.?\d*)', llm_output, re.IGNORECASE)
        if score_match:
            raw = float(score_match.group(1))
            return RAGEvaluator._normalize_score(raw)
        return 0.0
