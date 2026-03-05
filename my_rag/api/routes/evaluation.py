"""
评估接口

面试考点：
- 自动化评估的 API 化：配置变更后一键触发评估
- 评估结果的结构化返回
- LLM 辅助生成评估数据集：从知识库 Chunk 自动生成 QA 对
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter

from my_rag.api.schemas.common import APIResponse
from my_rag.config.settings import settings
from my_rag.core.dependencies import get_embedding, get_llm, get_rag_pipeline
from my_rag.evaluation.dataset_generator import DatasetGenerator, GenerationConfig
from my_rag.evaluation.metrics import RAGEvaluator
from my_rag.evaluation.evaluator import PipelineEvaluator
from my_rag.utils.logger import get_logger

router = APIRouter(prefix="/evaluations")
logger = get_logger(__name__)


class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, description="评估问题")
    knowledge_base_id: str = Field(..., description="知识库 ID")
    ground_truth: str = Field(default="", description="标准答案（可选）")


class EvalScores(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    overall: float = 0.0


class EvalResponse(BaseModel):
    question: str
    answer: str
    scores: EvalScores
    contexts: list[str] = []


@router.post("", response_model=APIResponse[EvalResponse])
async def run_evaluation(body: EvalRequest):
    """对单条问题执行 RAG + 评估"""
    pipeline = get_rag_pipeline()
    evaluator_metrics = RAGEvaluator(llm=get_llm(), embedding=get_embedding())
    evaluator = PipelineEvaluator(pipeline=pipeline, evaluator=evaluator_metrics)

    result = await evaluator.evaluate_single(
        question=body.question,
        knowledge_base_id=body.knowledge_base_id,
        ground_truth=body.ground_truth,
    )

    from my_rag.utils.metrics import EVAL_SCORE
    EVAL_SCORE.labels(metric="faithfulness").set(result.faithfulness)
    EVAL_SCORE.labels(metric="answer_relevancy").set(result.answer_relevancy)
    EVAL_SCORE.labels(metric="context_recall").set(result.context_recall)
    EVAL_SCORE.labels(metric="answer_correctness").set(result.answer_correctness)

    return APIResponse(data=EvalResponse(
        question=result.question,
        answer=result.answer,
        scores=EvalScores(
            faithfulness=round(result.faithfulness, 4),
            answer_relevancy=round(result.answer_relevancy, 4),
            context_recall=round(result.context_recall, 4),
            answer_correctness=round(result.answer_correctness, 4),
            overall=round(result.overall_score, 4),
        ),
        contexts=result.contexts,
    ))


# ── LLM 辅助生成评估数据集 ──────────────────────────────────────────


class DatasetGenerateRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="知识库 ID")
    dataset_name: str = Field(default="", description="数据集名称（留空则自动生成）")
    num_questions_per_chunk: int = Field(default=2, ge=1, le=10, description="每个 Chunk 生成的问题数")
    max_chunks: int = Field(default=50, ge=1, le=500, description="最大采样 Chunk 数")
    sampling_strategy: str = Field(default="diverse", description="采样策略: random / diverse / sequential")
    include_multi_chunk: bool = Field(default=True, description="是否生成跨 Chunk 综合性问题")
    save_to_file: bool = Field(default=True, description="是否保存到文件")


class DatasetSampleResponse(BaseModel):
    question: str
    ground_truth: str


class DatasetGenerateResponse(BaseModel):
    name: str
    total_samples: int
    file_path: str | None = None
    knowledge_base_id: str = ""
    samples: list[DatasetSampleResponse]


@router.post("/generate-dataset", response_model=APIResponse[DatasetGenerateResponse])
async def generate_eval_dataset(body: DatasetGenerateRequest):
    """LLM 辅助生成评估数据集：从知识库 Chunk 自动生成 QA 对"""
    config = GenerationConfig(
        num_questions_per_chunk=body.num_questions_per_chunk,
        max_chunks=body.max_chunks,
        sampling_strategy=body.sampling_strategy,
        include_multi_chunk_questions=body.include_multi_chunk,
    )
    generator = DatasetGenerator(llm=get_llm(), config=config)

    dataset = await generator.generate(
        knowledge_base_id=body.knowledge_base_id,
        dataset_name=body.dataset_name,
    )

    file_path = None
    if body.save_to_file:
        save_dir = settings.data_dir / "eval_datasets"
        file_path = str(save_dir / f"{dataset.name}.json")
        dataset.save(file_path)

    return APIResponse(data=DatasetGenerateResponse(
        name=dataset.name,
        total_samples=len(dataset.samples),
        file_path=file_path,
        knowledge_base_id=body.knowledge_base_id,
        samples=[
            DatasetSampleResponse(question=s.question, ground_truth=s.ground_truth)
            for s in dataset.samples
        ],
    ))


# ── 数据集管理 ────────────────────────────────────────────────────────


class DatasetListItem(BaseModel):
    name: str
    file_path: str
    total_samples: int
    file_size_kb: float


@router.get("/datasets", response_model=APIResponse[list[DatasetListItem]])
async def list_datasets():
    """列出所有已保存的评估数据集"""
    from my_rag.evaluation.dataset import EvalDataset

    save_dir = settings.data_dir / "eval_datasets"
    if not save_dir.exists():
        return APIResponse(data=[])

    items = []
    for fp in sorted(save_dir.glob("*.json")):
        try:
            ds = EvalDataset.load(fp)
            items.append(DatasetListItem(
                name=ds.name,
                file_path=str(fp),
                total_samples=len(ds.samples),
                file_size_kb=round(fp.stat().st_size / 1024, 1),
            ))
        except Exception as e:
            logger.warning("dataset_list_skip", path=str(fp), error=str(e))
    return APIResponse(data=items)


@router.get("/datasets/{dataset_name}", response_model=APIResponse[DatasetGenerateResponse])
async def get_dataset(dataset_name: str):
    """加载指定名称的评估数据集"""
    from pathlib import Path
    from my_rag.evaluation.dataset import EvalDataset

    fp = settings.data_dir / "eval_datasets" / f"{dataset_name}.json"
    if not fp.exists():
        return APIResponse(code=404, message=f"数据集 '{dataset_name}' 不存在", data=None)

    ds = EvalDataset.load(fp)
    kb_ids = {s.knowledge_base_id for s in ds.samples if s.knowledge_base_id}
    kb_id = kb_ids.pop() if len(kb_ids) == 1 else ""
    return APIResponse(data=DatasetGenerateResponse(
        name=ds.name,
        total_samples=len(ds.samples),
        file_path=str(fp),
        knowledge_base_id=kb_id,
        samples=[
            DatasetSampleResponse(question=s.question, ground_truth=s.ground_truth)
            for s in ds.samples
        ],
    ))


@router.delete("/datasets/{dataset_name}", response_model=APIResponse)
async def delete_dataset(dataset_name: str):
    """删除指定名称的评估数据集"""
    from pathlib import Path

    fp = settings.data_dir / "eval_datasets" / f"{dataset_name}.json"
    if not fp.exists():
        return APIResponse(code=404, message=f"数据集 '{dataset_name}' 不存在")

    fp.unlink()
    logger.info("dataset_deleted", name=dataset_name, path=str(fp))
    return APIResponse(message=f"数据集 '{dataset_name}' 已删除")


# ── 批量评估 ──────────────────────────────────────────────────────────


class BatchEvalSample(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(default="")


class BatchEvalRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="知识库 ID")
    samples: list[BatchEvalSample] = Field(..., min_length=1, description="评估样本列表")


class BatchEvalResultItem(BaseModel):
    question: str
    answer: str
    ground_truth: str
    scores: EvalScores
    contexts: list[str] = []


class BatchEvalSummary(BaseModel):
    total_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_answer_correctness: float


class BatchEvalResponse(BaseModel):
    summary: BatchEvalSummary
    results: list[BatchEvalResultItem]


@router.post("/batch", response_model=APIResponse[BatchEvalResponse])
async def run_batch_evaluation(body: BatchEvalRequest):
    """对多条样本批量执行 RAG + 评估，返回汇总报告"""
    pipeline = get_rag_pipeline()
    evaluator_metrics = RAGEvaluator(llm=get_llm(), embedding=get_embedding())
    evaluator = PipelineEvaluator(pipeline=pipeline, evaluator=evaluator_metrics)

    from my_rag.evaluation.dataset import EvalDataset
    dataset = EvalDataset(name="batch_eval")
    for s in body.samples:
        dataset.add(
            question=s.question,
            ground_truth=s.ground_truth,
            knowledge_base_id=body.knowledge_base_id,
        )

    report = await evaluator.evaluate_dataset(dataset, knowledge_base_id=body.knowledge_base_id)

    results = []
    for r in report.results:
        results.append(BatchEvalResultItem(
            question=r.question,
            answer=r.answer,
            ground_truth=r.ground_truth,
            scores=EvalScores(
                faithfulness=round(r.faithfulness, 4),
                answer_relevancy=round(r.answer_relevancy, 4),
                context_recall=round(r.context_recall, 4),
                answer_correctness=round(r.answer_correctness, 4),
                overall=round(r.overall_score, 4),
            ),
            contexts=r.contexts,
        ))

    summary_data = report.summary()
    return APIResponse(data=BatchEvalResponse(
        summary=BatchEvalSummary(
            total_samples=summary_data["total_samples"],
            avg_faithfulness=summary_data["avg_faithfulness"],
            avg_answer_relevancy=summary_data["avg_answer_relevancy"],
            avg_context_recall=summary_data["avg_context_recall"],
            avg_answer_correctness=summary_data["avg_answer_correctness"],
        ),
        results=results,
    ))
