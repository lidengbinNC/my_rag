"""
评估接口

面试考点：
- 自动化评估的 API 化：配置变更后一键触发评估
- 评估结果的结构化返回
- LLM 辅助生成评估数据集：从知识库 Chunk 自动生成 QA 对
- 评估结果持久化：EvalRun + EvalResultItem 落库，支持历史对比
"""

import json

from pydantic import BaseModel, Field
from fastapi import APIRouter, BackgroundTasks, Query

from my_rag.api.schemas.common import APIResponse
from my_rag.config.settings import settings
from my_rag.core.dependencies import get_embedding, get_llm, get_rag_pipeline
from my_rag.evaluation.dataset_generator import DatasetGenerator, GenerationConfig
from my_rag.evaluation.metrics import RAGEvaluator
from my_rag.evaluation.evaluator import PipelineEvaluator
from my_rag.infrastructure.database import async_session_factory, EvalRun, EvalResultItem
from my_rag.utils.logger import get_logger

router = APIRouter(prefix="/evaluations")
logger = get_logger(__name__)


# ── 持久化辅助函数 ────────────────────────────────────────────────────

async def _persist_single_result(
    result,
    knowledge_base_id: str,
    run_name: str = "",
) -> str:
    """将单条评估结果写入 eval_runs + eval_result_items，返回 run_id"""
    async with async_session_factory() as session:
        run = EvalRun(
            name=run_name or f"single:{result.question[:40]}",
            knowledge_base_id=knowledge_base_id,
            run_type="single",
            total_samples=1,
            avg_faithfulness=result.faithfulness,
            avg_answer_relevancy=result.answer_relevancy,
            avg_context_recall=result.context_recall,
            avg_answer_correctness=result.answer_correctness,
            avg_exact_match=result.exact_match,
            avg_token_f1=result.token_f1,
        )
        session.add(run)
        await session.flush()  # 获取 run.id

        item = EvalResultItem(
            run_id=run.id,
            question=result.question,
            answer=result.answer,
            ground_truth=result.ground_truth or "",
            contexts_json=json.dumps(result.contexts, ensure_ascii=False) if result.contexts else None,
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_recall=result.context_recall,
            answer_correctness=result.answer_correctness,
            overall_score=result.overall_score,
            exact_match=result.exact_match,
            token_f1=result.token_f1,
        )
        session.add(item)
        await session.commit()
        return run.id


async def _persist_batch_report(
    report,
    knowledge_base_id: str,
    dataset_name: str = "",
    run_name: str = "",
    config_snapshot: dict | None = None,
) -> str:
    """将批量评估报告写入 eval_runs + eval_result_items，返回 run_id"""
    summary = report.summary()
    async with async_session_factory() as session:
        run = EvalRun(
            name=run_name or f"batch:{dataset_name or 'manual'}",
            knowledge_base_id=knowledge_base_id,
            dataset_name=dataset_name,
            run_type="batch",
            total_samples=summary.get("total_samples", len(report.results)),
            avg_faithfulness=summary.get("avg_faithfulness"),
            avg_answer_relevancy=summary.get("avg_answer_relevancy"),
            avg_context_recall=summary.get("avg_context_recall"),
            avg_answer_correctness=summary.get("avg_answer_correctness"),
            avg_exact_match=summary.get("avg_exact_match"),
            avg_token_f1=summary.get("avg_token_f1"),
            config_snapshot=json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None,
        )
        session.add(run)
        await session.flush()

        for r in report.results:
            item = EvalResultItem(
                run_id=run.id,
                question=r.question,
                answer=r.answer,
                ground_truth=r.ground_truth or "",
                contexts_json=json.dumps(r.contexts, ensure_ascii=False) if r.contexts else None,
                faithfulness=r.faithfulness,
                answer_relevancy=r.answer_relevancy,
                context_recall=r.context_recall,
                answer_correctness=r.answer_correctness,
                overall_score=r.overall_score,
                exact_match=r.exact_match,
                token_f1=r.token_f1,
            )
            session.add(item)

        await session.commit()
        return run.id


class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, description="评估问题")
    knowledge_base_id: str = Field(..., description="知识库 ID")
    ground_truth: str = Field(default="", description="标准答案（可选）")
    supporting_titles:list[str] = Field(..., description="相关文件title")


class EvalScores(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    overall: float = 0.0
    # HotpotQA 官方指标（有 ground_truth 时自动计算，-1 表示未计算）
    exact_match: float = -1.0
    token_f1: float = -1.0


class EvalResponse(BaseModel):
    question: str
    answer: str
    scores: EvalScores
    contexts: list[str] = []
    run_id: str | None = None


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
        supporting_titles=body.supporting_titles
    )

    from my_rag.utils.metrics import EVAL_SCORE
    # 指标可能为 None（未计算），Prometheus gauge 只接受数值，未计算时上报 -1 以示区分
    EVAL_SCORE.labels(metric="faithfulness").set(result.faithfulness if result.faithfulness is not None else -1)
    EVAL_SCORE.labels(metric="answer_relevancy").set(result.answer_relevancy if result.answer_relevancy is not None else -1)
    EVAL_SCORE.labels(metric="context_recall").set(result.context_recall if result.context_recall is not None else -1)
    EVAL_SCORE.labels(metric="answer_correctness").set(result.answer_correctness if result.answer_correctness is not None else -1)

    # 持久化到 MySQL（异步，不阻塞响应）
    try:
        run_id = await _persist_single_result(result, knowledge_base_id=body.knowledge_base_id)
        logger.info("eval_result_persisted", run_id=run_id, run_type="single")
    except Exception as exc:
        logger.warning("eval_persist_failed", error=str(exc))
        run_id = None

    def _score(v: float | None) -> float:
        return round(v, 4) if v is not None else 0.0

    return APIResponse(data=EvalResponse(
        question=result.question,
        answer=result.answer,
        scores=EvalScores(
            faithfulness=_score(result.faithfulness),
            answer_relevancy=_score(result.answer_relevancy),
            context_recall=_score(result.context_recall),
            answer_correctness=_score(result.answer_correctness),
            overall=round(result.overall_score, 4),
            exact_match=_score(result.exact_match) if result.exact_match is not None else -1.0,
            token_f1=_score(result.token_f1) if result.token_f1 is not None else -1.0,
        ),
        contexts=result.contexts,
        run_id=run_id,
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
    metadata: dict = {}


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
            DatasetSampleResponse(question=s.question, ground_truth=s.ground_truth, metadata=s.metadata)
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
            DatasetSampleResponse(question=s.question, ground_truth=s.ground_truth, metadata=s.metadata)
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


# ── HotpotQA 数据集导入 ───────────────────────────────────────────────


class HotpotQAImportRequest(BaseModel):
    num_samples: int = Field(default=20, ge=1, le=500, description="拉取样本数量")
    split: str = Field(default="train", description="数据集分片：train / validation")
    config_name: str = Field(default="distractor", description="数据集配置：distractor / fullwiki")
    only_supporting: bool = Field(
        default=False,
        description="True：只保留支撑文章（更干净）；False：保留全部 10 篇含干扰项（更真实）",
    )
    export_docx: bool = Field(default=True, description="是否将 context 文章导出为 docx 文件")
    merge_docx: bool = Field(
        default=False,
        description="True：所有文章合并为一个 docx；False：每篇文章单独一个 docx（推荐）",
    )
    dataset_name: str = Field(default="", description="数据集名称，留空则自动生成")
    knowledge_base_id: str = Field(default="", description="关联的知识库 ID（指定后自动将 context 文章写入 MySQL + Milvus）")
    auto_ingest: bool = Field(default=True, description="是否自动将 context 文章写入知识库（需指定 knowledge_base_id）")


class HotpotQAImportResponse(BaseModel):
    dataset_name: str
    total_samples: int
    eval_json_path: str
    raw_json_path: str | None = None   # 完整原始 JSON 路径（含 context/supporting_facts）
    docx_dir: str | None = None
    docx_files: list[str] = []
    total_articles: int = 0
    ingest_task_id: str | None = None  # 后台 ingest 任务 ID，用于轮询进度
    ingest_total: int = 0              # 需要 ingest 的文章总数
    samples: list[DatasetSampleResponse] = []


@router.post("/import-hotpotqa", response_model=APIResponse[HotpotQAImportResponse])
async def import_hotpotqa_dataset(body: HotpotQAImportRequest, background_tasks: BackgroundTasks):
    """
    从 HuggingFace 拉取 HotpotQA 数据集并一键完成全流程

    完整流程：
    1. [同步，to_thread] 从 HuggingFace 拉取样本、生成 docx、保存 JSON
    2. 立即返回 HTTP 响应（含 ingest_task_id）
    3. [后台任务] 批量 Parse → Chunk → 合并 Embed → 写 Milvus/MySQL/BM25
       前端通过 GET /ingest-status/{task_id} 轮询进度

    面试考点：
    - 使用真实公开数据集避免 LLM 造数据的偏差
    - HotpotQA 多跳推理特性：测试 RAG 跨文档检索能力
    - 后台任务模式：耗时操作（embed）不阻塞 HTTP 响应，用户体验更好
    - 批量 embed：所有文章 chunk 合并一次推理，消除 BGE-M3 固定启动开销
    """
    import asyncio
    import uuid
    from datetime import datetime
    from pathlib import Path
    from my_rag.evaluation.hotpotqa_loader import HotpotQALoader, HotpotQALoaderConfig

    dataset_name = body.dataset_name or f"hotpotqa_{body.config_name}_{body.num_samples}"

    docx_dir = None
    if body.export_docx:
        docx_dir = str(settings.data_dir / "hotpotqa_docs" / dataset_name)

    raw_json_dir = str(settings.data_dir / "eval_datasets" / "raw")

    cfg = HotpotQALoaderConfig(
        split=body.split,
        num_samples=body.num_samples,
        config_name=body.config_name,
        only_supporting=body.only_supporting,
        output_dir=docx_dir,
        raw_json_dir=raw_json_dir,
    )
    loader = HotpotQALoader(cfg)

    try:
        # load_dataset + docx 生成 + raw JSON 写入均为同步阻塞操作（磁盘 IO + HuggingFace 网络）
        # 用 asyncio.to_thread 放入线程池，避免阻塞 FastAPI event loop
        samples, eval_dataset = await asyncio.to_thread(loader.load)
    except Exception as e:
        logger.error("hotpotqa_load_failed", error=str(e))
        return APIResponse(code=500, message=f"拉取 HotpotQA 数据集失败: {e}", data=None)

    # 填充 knowledge_base_id
    if body.knowledge_base_id:
        for s in eval_dataset.samples:
            if not s.knowledge_base_id:
                s.knowledge_base_id = body.knowledge_base_id

    eval_dataset.name = dataset_name

    save_dir = settings.data_dir / "eval_datasets"
    eval_json_path = str(save_dir / f"{dataset_name}.json")
    # eval_dataset.save 也是同步文件写入，放入线程池
    await asyncio.to_thread(eval_dataset.save, eval_json_path)

    raw_json_path = str(Path(raw_json_dir) / f"{dataset_name}_raw.json")

    # 统计 docx 文件
    docx_files: list[str] = []
    total_articles = 0
    if docx_dir:
        docx_path = Path(docx_dir)
        if docx_path.exists():
            docx_files = [str(p) for p in sorted(docx_path.glob("*.docx"))]
            total_articles = len(docx_files)

    # ── 自动 ingest：注册文档到 MySQL，后台批量 chunk + embed + 写 Milvus ──
    ingest_task_id = None
    ingest_total = 0

    if body.auto_ingest and body.knowledge_base_id and docx_files:
        from my_rag.infrastructure.database import KnowledgeBase, Document
        from my_rag.infrastructure.database.session import async_session_factory
        from my_rag.core.document_pipeline import batch_ingest_documents

        async with async_session_factory() as db:
            kb = await db.get(KnowledgeBase, body.knowledge_base_id)
            if not kb:
                logger.warning("hotpotqa_ingest_skip_no_kb", kb_id=body.knowledge_base_id)
            else:
                # 同步文件复制放入线程池
                upload_dir = settings.upload_path / body.knowledge_base_id
                await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)

                doc_ids: list[str] = []
                for fp_str in docx_files:
                    fp = Path(fp_str)
                    try:
                        content = await asyncio.to_thread(fp.read_bytes)
                        if len(content) > settings.storage.max_file_size:
                            logger.warning("hotpotqa_ingest_skip_too_large", filename=fp.name)
                            continue

                        doc_id = str(uuid.uuid4())
                        dest_path = upload_dir / f"{doc_id}.docx"
                        await asyncio.to_thread(dest_path.write_bytes, content)

                        doc = Document(
                            id=doc_id,
                            filename=fp.name,
                            file_path=str(dest_path),
                            file_size=len(content),
                            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            status="pending",
                            knowledge_base_id=body.knowledge_base_id,
                            created_at=datetime.now(),
                        )
                        db.add(doc)
                        doc_ids.append(doc_id)
                    except Exception as e:
                        logger.error("hotpotqa_ingest_register_failed", filename=fp.name, error=str(e))

                kb.document_count += len(doc_ids)
                kb.updated_at = datetime.now()
                await db.commit()

                if doc_ids:
                    # 后台任务：立即返回响应，embed/写库在后台执行
                    ingest_task_id = dataset_name
                    ingest_total = len(doc_ids)
                    background_tasks.add_task(
                        batch_ingest_documents,
                        doc_ids=doc_ids,
                        task_id=ingest_task_id,
                        concurrency=4,
                    )
                    logger.info(
                        "hotpotqa_ingest_queued",
                        task_id=ingest_task_id,
                        total=ingest_total,
                    )

    logger.info(
        "hotpotqa_imported",
        dataset_name=dataset_name,
        total_samples=len(eval_dataset.samples),
        total_articles=total_articles,
        ingest_task_id=ingest_task_id,
        eval_json_path=eval_json_path,
        raw_json_path=raw_json_path,
    )

    return APIResponse(data=HotpotQAImportResponse(
        dataset_name=dataset_name,
        total_samples=len(eval_dataset.samples),
        eval_json_path=eval_json_path,
        raw_json_path=raw_json_path,
        docx_dir=docx_dir,
        docx_files=docx_files,
        total_articles=total_articles,
        ingest_task_id=ingest_task_id,
        ingest_total=ingest_total,
        samples=[
            DatasetSampleResponse(question=s.question, ground_truth=s.ground_truth, metadata=s.metadata)
            for s in eval_dataset.samples[:10]
        ],
    ))


# ── Ingest 进度查询 ───────────────────────────────────────────────────


class IngestStatusResponse(BaseModel):
    task_id: str
    total: int
    done: int
    failed: int
    status: str          # running | completed | failed
    elapsed_s: float
    error: str | None = None


@router.get("/ingest-status/{task_id}", response_model=APIResponse[IngestStatusResponse])
async def get_ingest_status(task_id: str):
    """
    查询后台 ingest 任务的进度

    前端在 import-hotpotqa 返回 ingest_task_id 后，每隔 2 秒轮询此接口，
    直到 status == 'completed' 或 'failed'。
    """
    from my_rag.core.document_pipeline import get_ingest_progress

    progress = get_ingest_progress(task_id)
    if progress is None:
        return APIResponse(code=404, message=f"任务 '{task_id}' 不存在或已过期", data=None)

    return APIResponse(data=IngestStatusResponse(
        task_id=progress.task_id,
        total=progress.total,
        done=progress.done,
        failed=progress.failed,
        status=progress.status,
        elapsed_s=progress.elapsed_s,
        error=progress.error,
    ))


# ── HotpotQA docx 批量上传到知识库 ───────────────────────────────────


class HotpotQAUploadRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="目标知识库 ID")
    docx_dir: str = Field(..., description="docx 文件所在目录（服务端路径）")


class HotpotQAUploadResponse(BaseModel):
    total: int
    success: int
    failed: int
    skipped: int
    results: list[dict] = []


@router.post("/upload-hotpotqa-docs", response_model=APIResponse[HotpotQAUploadResponse])
async def upload_hotpotqa_docs(body: HotpotQAUploadRequest):
    """
    将服务端 docx 目录中的文件批量上传到指定知识库

    适用场景：import-hotpotqa 生成 docx 文件后，一键将所有文章上传到知识库，
    无需用户手动下载再上传。

    面试考点：
    - 服务端文件直接处理，避免大文件在客户端和服务端之间来回传输
    - 与前端批量上传接口的区别：这里文件已在服务端，直接调用 process_document
    """
    from pathlib import Path
    from datetime import datetime
    import uuid
    from my_rag.infrastructure.database import KnowledgeBase, Document, get_db
    from my_rag.infrastructure.database.session import async_session_factory
    from my_rag.core.document_pipeline import process_document

    docx_path = Path(body.docx_dir)
    if not docx_path.exists() or not docx_path.is_dir():
        return APIResponse(code=400, message=f"目录不存在: {body.docx_dir}", data=None)

    docx_files = sorted(docx_path.glob("*.docx"))
    if not docx_files:
        return APIResponse(code=400, message="目录中没有 docx 文件", data=None)

    results = []
    success_count = 0
    failed_count = 0

    async with async_session_factory() as db:
        kb = await db.get(KnowledgeBase, body.knowledge_base_id)
        if not kb:
            return APIResponse(code=404, message="知识库不存在", data=None)

        for fp in docx_files:
            try:
                content = fp.read_bytes()
                if len(content) > settings.storage.max_file_size:
                    results.append({"filename": fp.name, "status": "skipped", "error": "文件过大"})
                    continue

                upload_dir = settings.upload_path / body.knowledge_base_id
                upload_dir.mkdir(parents=True, exist_ok=True)

                doc_id = str(uuid.uuid4())
                dest_path = upload_dir / f"{doc_id}.docx"
                dest_path.write_bytes(content)

                doc = Document(
                    id=doc_id,
                    filename=fp.name,
                    file_path=str(dest_path),
                    file_size=len(content),
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    status="pending",
                    knowledge_base_id=body.knowledge_base_id,
                    created_at=datetime.now(),
                )
                db.add(doc)
                kb.document_count += 1
                kb.updated_at = datetime.now()
                await db.commit()

                await process_document(doc_id)
                results.append({"filename": fp.name, "status": "success", "doc_id": doc_id})
                success_count += 1

            except Exception as e:
                logger.error("hotpotqa_doc_upload_failed", filename=fp.name, error=str(e))
                results.append({"filename": fp.name, "status": "failed", "error": str(e)})
                failed_count += 1

    logger.info(
        "hotpotqa_docs_uploaded",
        kb_id=body.knowledge_base_id,
        total=len(docx_files),
        success=success_count,
        failed=failed_count,
    )

    return APIResponse(data=HotpotQAUploadResponse(
        total=len(docx_files),
        success=success_count,
        failed=failed_count,
        skipped=len(docx_files) - success_count - failed_count,
        results=results,
    ))


# ── 批量评估 ──────────────────────────────────────────────────────────


class BatchEvalSample(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(default="")


class BatchEvalRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="知识库 ID")
    samples: list[BatchEvalSample] = Field(..., min_length=1, description="评估样本列表")
    run_name: str = Field(default="", description="本次评估的名称（留空则自动生成）")
    dataset_name: str = Field(default="", description="关联的数据集名称")


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
    run_id: str | None = None


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

    def _score(v: float | None) -> float:
        return round(v, 4) if v is not None else 0.0

    results = []
    for r in report.results:
        results.append(BatchEvalResultItem(
            question=r.question,
            answer=r.answer,
            ground_truth=r.ground_truth,
            scores=EvalScores(
                faithfulness=_score(r.faithfulness),
                answer_relevancy=_score(r.answer_relevancy),
                context_recall=_score(r.context_recall),
                answer_correctness=_score(r.answer_correctness),
                overall=round(r.overall_score, 4),
                exact_match=_score(r.exact_match) if r.exact_match is not None else -1.0,
                token_f1=_score(r.token_f1) if r.token_f1 is not None else -1.0,
            ),
            contexts=r.contexts,
        ))

    # 持久化到 MySQL
    try:
        run_id = await _persist_batch_report(
            report,
            knowledge_base_id=body.knowledge_base_id,
            dataset_name=body.dataset_name,
            run_name=body.run_name,
        )
        logger.info("eval_result_persisted", run_id=run_id, run_type="batch", total=len(report.results))
    except Exception as exc:
        logger.warning("eval_persist_failed", error=str(exc))
        run_id = None

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
        run_id=run_id,
    ))


# ── 评估历史查询接口 ──────────────────────────────────────────────────


class EvalRunSummary(BaseModel):
    """EvalRun 列表摘要（不含明细行，减少传输量）"""
    run_id: str
    name: str
    knowledge_base_id: str
    dataset_name: str
    run_type: str
    total_samples: int
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_context_recall: float | None
    avg_answer_correctness: float | None
    avg_exact_match: float | None
    avg_token_f1: float | None
    created_at: str


class EvalRunDetail(EvalRunSummary):
    """EvalRun 详情（含所有明细行）"""
    items: list[dict]


class EvalRunCompare(BaseModel):
    """两条 EvalRun 的对比结果"""
    run_a: EvalRunSummary
    run_b: EvalRunSummary
    # 逐指标差值：run_b - run_a（正数表示 b 更好）
    delta: dict[str, float | None]


@router.get("/history", response_model=APIResponse[list[EvalRunSummary]])
async def list_eval_runs(
    knowledge_base_id: str | None = Query(default=None, description="按知识库过滤"),
    run_type: str | None = Query(default=None, description="single | batch"),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """查询评估历史列表，按创建时间倒序"""
    from sqlalchemy import select, desc
    async with async_session_factory() as session:
        stmt = select(EvalRun).order_by(desc(EvalRun.created_at)).offset(offset).limit(limit)
        if knowledge_base_id:
            stmt = stmt.where(EvalRun.knowledge_base_id == knowledge_base_id)
        if run_type:
            stmt = stmt.where(EvalRun.run_type == run_type)
        rows = (await session.execute(stmt)).scalars().all()

    def _to_summary(r: EvalRun) -> EvalRunSummary:
        return EvalRunSummary(
            run_id=r.id,
            name=r.name,
            knowledge_base_id=r.knowledge_base_id,
            dataset_name=r.dataset_name,
            run_type=r.run_type,
            total_samples=r.total_samples,
            avg_faithfulness=round(r.avg_faithfulness, 4) if r.avg_faithfulness is not None else None,
            avg_answer_relevancy=round(r.avg_answer_relevancy, 4) if r.avg_answer_relevancy is not None else None,
            avg_context_recall=round(r.avg_context_recall, 4) if r.avg_context_recall is not None else None,
            avg_answer_correctness=round(r.avg_answer_correctness, 4) if r.avg_answer_correctness is not None else None,
            avg_exact_match=round(r.avg_exact_match, 4) if r.avg_exact_match is not None else None,
            avg_token_f1=round(r.avg_token_f1, 4) if r.avg_token_f1 is not None else None,
            created_at=r.created_at.isoformat(),
        )

    return APIResponse(data=[_to_summary(r) for r in rows])


@router.get("/history/compare/{run_id_a}/{run_id_b}", response_model=APIResponse[EvalRunCompare])
async def compare_eval_runs(run_id_a: str, run_id_b: str):
    """对比两次评估的汇总指标，返回各指标差值（run_b - run_a）"""
    from sqlalchemy import select
    async with async_session_factory() as session:
        stmt = select(EvalRun).where(EvalRun.id.in_([run_id_a, run_id_b]))
        runs = {r.id: r for r in (await session.execute(stmt)).scalars().all()}

    if run_id_a not in runs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"EvalRun {run_id_a} not found")
    if run_id_b not in runs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"EvalRun {run_id_b} not found")

    def _to_summary(r: EvalRun) -> EvalRunSummary:
        return EvalRunSummary(
            run_id=r.id,
            name=r.name,
            knowledge_base_id=r.knowledge_base_id,
            dataset_name=r.dataset_name,
            run_type=r.run_type,
            total_samples=r.total_samples,
            avg_faithfulness=round(r.avg_faithfulness, 4) if r.avg_faithfulness is not None else None,
            avg_answer_relevancy=round(r.avg_answer_relevancy, 4) if r.avg_answer_relevancy is not None else None,
            avg_context_recall=round(r.avg_context_recall, 4) if r.avg_context_recall is not None else None,
            avg_answer_correctness=round(r.avg_answer_correctness, 4) if r.avg_answer_correctness is not None else None,
            avg_exact_match=round(r.avg_exact_match, 4) if r.avg_exact_match is not None else None,
            avg_token_f1=round(r.avg_token_f1, 4) if r.avg_token_f1 is not None else None,
            created_at=r.created_at.isoformat(),
        )

    ra, rb = runs[run_id_a], runs[run_id_b]

    def _delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 4)

    delta = {
        "avg_faithfulness": _delta(ra.avg_faithfulness, rb.avg_faithfulness),
        "avg_answer_relevancy": _delta(ra.avg_answer_relevancy, rb.avg_answer_relevancy),
        "avg_context_recall": _delta(ra.avg_context_recall, rb.avg_context_recall),
        "avg_answer_correctness": _delta(ra.avg_answer_correctness, rb.avg_answer_correctness),
        "avg_exact_match": _delta(ra.avg_exact_match, rb.avg_exact_match),
        "avg_token_f1": _delta(ra.avg_token_f1, rb.avg_token_f1),
    }

    return APIResponse(data=EvalRunCompare(
        run_a=_to_summary(ra),
        run_b=_to_summary(rb),
        delta=delta,
    ))


@router.get("/history/{run_id}", response_model=APIResponse[EvalRunDetail])
async def get_eval_run_detail(run_id: str):
    """查询单次评估的完整明细（含每条问答的所有指标）"""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    async with async_session_factory() as session:
        stmt = (
            select(EvalRun)
            .where(EvalRun.id == run_id)
            .options(selectinload(EvalRun.items))
        )
        run = (await session.execute(stmt)).scalar_one_or_none()

    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"EvalRun {run_id} not found")

    items = []
    for it in run.items:
        items.append({
            "id": it.id,
            "question": it.question,
            "answer": it.answer,
            "ground_truth": it.ground_truth,
            "contexts": json.loads(it.contexts_json) if it.contexts_json else [],
            "faithfulness": it.faithfulness,
            "answer_relevancy": it.answer_relevancy,
            "context_recall": it.context_recall,
            "answer_correctness": it.answer_correctness,
            "overall_score": it.overall_score,
            "exact_match": it.exact_match,
            "token_f1": it.token_f1,
            "created_at": it.created_at.isoformat(),
        })

    detail = EvalRunDetail(
        run_id=run.id,
        name=run.name,
        knowledge_base_id=run.knowledge_base_id,
        dataset_name=run.dataset_name,
        run_type=run.run_type,
        total_samples=run.total_samples,
        avg_faithfulness=round(run.avg_faithfulness, 4) if run.avg_faithfulness is not None else None,
        avg_answer_relevancy=round(run.avg_answer_relevancy, 4) if run.avg_answer_relevancy is not None else None,
        avg_context_recall=round(run.avg_context_recall, 4) if run.avg_context_recall is not None else None,
        avg_answer_correctness=round(run.avg_answer_correctness, 4) if run.avg_answer_correctness is not None else None,
        avg_exact_match=round(run.avg_exact_match, 4) if run.avg_exact_match is not None else None,
        avg_token_f1=round(run.avg_token_f1, 4) if run.avg_token_f1 is not None else None,
        created_at=run.created_at.isoformat(),
        items=items,
    )
    return APIResponse(data=detail)


@router.delete("/history/{run_id}", response_model=APIResponse[dict])
async def delete_eval_run(run_id: str):
    """删除一条评估历史记录（级联删除明细行）"""
    from sqlalchemy import select
    async with async_session_factory() as session:
        run = (await session.execute(
            select(EvalRun).where(EvalRun.id == run_id)
        )).scalar_one_or_none()
        if run is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"EvalRun {run_id} not found")
        await session.delete(run)
        await session.commit()

    return APIResponse(data={"deleted": run_id})
