"""Tests for RAGAS-style evaluation metrics and dataset management."""

import json
import tempfile
from pathlib import Path

import pytest

from my_rag.evaluation.dataset import EvalDataset, EvalSample
from my_rag.evaluation.metrics import EvaluationReport, EvaluationResult, RAGEvaluator
from tests.conftest import FakeEmbedding, FakeLLM


# ── EvalDataset ──────────────────────────────────────────────────────


class TestEvalDataset:
    def test_add_and_access(self):
        ds = EvalDataset(name="test")
        ds.add("q1", "a1", "kb_1")
        assert len(ds.samples) == 1
        assert ds.samples[0].question == "q1"

    def test_save_and_load(self, tmp_path: Path):
        ds = EvalDataset(name="test_ds")
        ds.add("问题1", "答案1", "kb_1")
        ds.add("问题2", "答案2", "kb_2")

        file_path = tmp_path / "eval.json"
        ds.save(file_path)
        assert file_path.exists()

        loaded = EvalDataset.load(file_path)
        assert loaded.name == "test_ds"
        assert len(loaded.samples) == 2
        assert loaded.samples[0].question == "问题1"
        assert loaded.samples[1].ground_truth == "答案2"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        ds = EvalDataset(name="nested")
        ds.add("q", "a")
        nested_path = tmp_path / "deep" / "nested" / "eval.json"
        ds.save(nested_path)
        assert nested_path.exists()


# ── EvaluationResult ─────────────────────────────────────────────────


class TestEvaluationResult:
    def test_overall_score_average(self):
        r = EvaluationResult(
            question="q", answer="a",
            faithfulness=0.8, answer_relevancy=0.6,
            context_precision=0.0, context_recall=0.4,
        )
        # only non-zero scores included: (0.8 + 0.6 + 0.4) / 3
        assert r.overall_score == pytest.approx(0.6, abs=0.01)

    def test_overall_score_all_zero(self):
        r = EvaluationResult(question="q", answer="a")
        assert r.overall_score == 0.0


# ── EvaluationReport ─────────────────────────────────────────────────


class TestEvaluationReport:
    def test_summary_with_results(self):
        report = EvaluationReport(results=[
            EvaluationResult(question="q1", answer="a1", faithfulness=0.8, answer_relevancy=0.7),
            EvaluationResult(question="q2", answer="a2", faithfulness=0.6, answer_relevancy=0.9),
        ])
        summary = report.summary()
        assert summary["total_samples"] == 2
        assert summary["avg_faithfulness"] == pytest.approx(0.7, abs=0.01)
        assert summary["avg_answer_relevancy"] == pytest.approx(0.8, abs=0.01)

    def test_summary_empty(self):
        report = EvaluationReport()
        summary = report.summary()
        assert summary["total_samples"] == 0
        assert summary["avg_faithfulness"] == 0.0


# ── RAGEvaluator._extract_score ─────────────────────────────────────


class TestScoreExtraction:
    def test_extract_from_json(self):
        output = '{"total_statements": 5, "supported_statements": 4, "score": 0.80}'
        assert RAGEvaluator._extract_score(output) == pytest.approx(0.80)

    def test_extract_from_json_with_surrounding_text(self):
        output = 'Some text {"score": 0.75} more text'
        assert RAGEvaluator._extract_score(output) == pytest.approx(0.75)

    def test_extract_from_plain_text(self):
        output = "score: 0.65"
        assert RAGEvaluator._extract_score(output) == pytest.approx(0.65)

    def test_normalize_percentage_score(self):
        output = '{"score": 85}'
        assert RAGEvaluator._extract_score(output) == pytest.approx(0.85)

    def test_no_score_returns_zero(self):
        assert RAGEvaluator._extract_score("no score here") == 0.0

    def test_clamp_to_one(self):
        output = '{"score": 1.5}'
        score = RAGEvaluator._extract_score(output)
        assert score <= 1.0

    def test_normalize_score_boundary(self):
        assert RAGEvaluator._normalize_score(0.95) == 0.95
        assert RAGEvaluator._normalize_score(95.0) == 0.95
        assert RAGEvaluator._normalize_score(-0.5) == 0.0
        assert RAGEvaluator._normalize_score(150.0) == 1.0


# ── RAGEvaluator (with FakeLLM) ─────────────────────────────────────


class TestRAGEvaluator:
    @pytest.mark.asyncio
    async def test_faithfulness(self):
        llm = FakeLLM(responses=['{"total_statements": 4, "supported_statements": 3, "score": 0.75}'])
        evaluator = RAGEvaluator(llm, FakeEmbedding(dim=4))
        score = await evaluator.evaluate_faithfulness("answer", ["doc1", "doc2"])
        assert score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_faithfulness_error_returns_zero(self):
        class ErrorLLM(FakeLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise RuntimeError("API error")

        evaluator = RAGEvaluator(ErrorLLM(), FakeEmbedding(dim=4))
        score = await evaluator.evaluate_faithfulness("a", ["c"])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_relevancy(self):
        llm = FakeLLM(responses=["什么是RAG\nRAG的定义\nRAG系统"])
        evaluator = RAGEvaluator(llm, FakeEmbedding(dim=4))
        score = await evaluator.evaluate_relevancy("什么是RAG", "RAG 是一种技术")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_context_recall(self):
        llm = FakeLLM(responses=['{"total_points": 3, "covered_points": 2, "score": 0.67}'])
        evaluator = RAGEvaluator(llm, FakeEmbedding(dim=4))
        score = await evaluator.evaluate_context_recall("ground truth", ["doc"])
        assert score == pytest.approx(0.67)

    @pytest.mark.asyncio
    async def test_context_recall_no_ground_truth(self):
        evaluator = RAGEvaluator(FakeLLM(), FakeEmbedding(dim=4))
        assert await evaluator.evaluate_context_recall("", ["doc"]) == 0.0

    @pytest.mark.asyncio
    async def test_correctness(self):
        evaluator = RAGEvaluator(FakeLLM(), FakeEmbedding(dim=4))
        score = await evaluator.evaluate_correctness("answer text", "ground truth")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_correctness_no_ground_truth(self):
        evaluator = RAGEvaluator(FakeLLM(), FakeEmbedding(dim=4))
        assert await evaluator.evaluate_correctness("answer", "") == 0.0

    @pytest.mark.asyncio
    async def test_full_evaluate(self):
        llm = FakeLLM(responses=[
            '{"score": 0.80}',                   # faithfulness
            "q1\nq2\nq3",                         # relevancy
            '{"score": 0.70}',                    # recall
        ])
        evaluator = RAGEvaluator(llm, FakeEmbedding(dim=4))
        result = await evaluator.evaluate(
            question="q", answer="a",
            contexts=["c1"], ground_truth="gt",
        )
        assert result.faithfulness == pytest.approx(0.80)
        assert result.context_recall == pytest.approx(0.70)
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.answer_correctness <= 1.0
