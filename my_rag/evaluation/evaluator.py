"""
评估器：编排 RAG Pipeline + 评估指标

面试考点：
- 评估流程：对数据集中每条样本执行 RAG → 收集结果 → 计算指标 → 生成报告
- 自动化评估的价值：调参后能快速定量对比效果
- A/B 测试场景：切换分块策略/检索参数后，对比评估分数
"""

from my_rag.core.rag_pipeline import RAGPipeline
from my_rag.evaluation.dataset import EvalDataset
from my_rag.evaluation.metrics import EvaluationReport, EvaluationResult, RAGEvaluator
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineEvaluator:
    """端到端评估器：RAG Pipeline + RAGAS 指标"""

    def __init__(self, pipeline: RAGPipeline, evaluator: RAGEvaluator):
        self._pipeline = pipeline
        self._evaluator = evaluator

    async def evaluate_dataset(
        self,
        dataset: EvalDataset,
        knowledge_base_id: str = "",
    ) -> EvaluationReport:
        """对整个数据集运行评估"""
        report = EvaluationReport()

        for i, sample in enumerate(dataset.samples):
            kb_id = sample.knowledge_base_id or knowledge_base_id
            if not kb_id:
                logger.warning("skip_sample_no_kb", index=i, question=sample.question[:50])
                continue

            logger.info("evaluating_sample", index=i, total=len(dataset.samples), question=sample.question[:50])

            try:
                rag_result = await self._pipeline.run(
                    query=sample.question,
                    knowledge_base_id=kb_id,
                )

                contexts = [s.content for s in rag_result.sources]

                eval_result = await self._evaluator.evaluate(
                    question=sample.question,
                    answer=rag_result.answer,
                    contexts=contexts,
                    ground_truth=sample.ground_truth,
                )

                report.results.append(eval_result)

            except Exception as e:
                logger.error("evaluation_failed", index=i, error=str(e))
                report.results.append(EvaluationResult(
                    question=sample.question,
                    answer=f"ERROR: {e}",
                    ground_truth=sample.ground_truth,
                ))

        logger.info("evaluation_report", summary=report.summary())
        return report

    async def evaluate_single(
        self,
        question: str,
        knowledge_base_id: str,
        ground_truth: str = "",
    ) -> EvaluationResult:
        """评估单条问题"""
        rag_result = await self._pipeline.run(
            query=question,
            knowledge_base_id=knowledge_base_id,
        )

        contexts = [s.content for s in rag_result.sources]

        return await self._evaluator.evaluate(
            question=question,
            answer=rag_result.answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
