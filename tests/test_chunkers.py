"""Tests for chunking strategies (Fixed / Recursive / Parent-Child)."""

import pytest

from my_rag.domain.chunking.base import BaseChunker
from my_rag.domain.chunking.factory import ChunkerFactory
from my_rag.domain.chunking.fixed_chunker import FixedChunker
from my_rag.domain.chunking.parent_child_chunker import ParentChildChunker
from my_rag.domain.chunking.recursive_chunker import RecursiveChunker
from my_rag.utils.token_counter import count_tokens

LONG_TEXT = "这是一段很长的测试文本。" * 200  # ~1200+ tokens


# ── BaseChunker 校验 ─────────────────────────────────────────────────


class TestBaseChunkerValidation:
    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            FixedChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=50, chunk_overlap=60)


# ── FixedChunker ─────────────────────────────────────────────────────


class TestFixedChunker:
    def test_short_text_single_chunk(self):
        """短文本应只产生一个 chunk。"""
        chunker = FixedChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("Hello world")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["chunker"] == "fixed"

    def test_long_text_multiple_chunks(self):
        """长文本应被切成多个 chunk。"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 1

    def test_chunk_tokens_within_limit(self):
        """每个 chunk 的 token 数不超过 chunk_size。"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        for chunk in chunker.chunk(LONG_TEXT):
            assert chunk.token_count <= 100

    def test_chunk_indices_sequential(self):
        """chunk_index 应从 0 开始递增。"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(LONG_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_propagated(self):
        """传入的 metadata 应被继承。"""
        chunker = FixedChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("test text", metadata={"source": "test.txt"})
        assert chunks[0].metadata["source"] == "test.txt"

    def test_empty_text(self):
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("")
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].content == "")


# ── RecursiveChunker ─────────────────────────────────────────────────


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("短文本")
        assert len(chunks) == 1

    def test_splits_on_paragraphs(self):
        """段落分隔符应优先于句子分隔符。"""
        text = "第一段内容。" * 30 + "\n\n" + "第二段内容。" * 30
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_tokens_within_limit(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        for chunk in chunker.chunk(LONG_TEXT):
            token_count = count_tokens(chunk.content)
            assert token_count <= 100 * 1.2  # allow small overlap margin

    def test_no_empty_chunks(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_metadata_includes_chunker_tag(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("test")
        assert chunks[0].metadata["chunker"] == "recursive"

    def test_custom_separators(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5, separators=["|", ""])
        text = "AAA|BBB|CCC|DDD" * 20
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


# ── ParentChildChunker ───────────────────────────────────────────────


class TestParentChildChunker:
    def test_child_has_parent_content(self):
        """每个 child chunk 的 metadata 应包含 parent_content。"""
        chunker = ParentChildChunker(
            chunk_size=50, chunk_overlap=5,
            parent_chunk_size=200, parent_chunk_overlap=20,
        )
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "parent_content" in chunk.metadata
            assert len(chunk.metadata["parent_content"]) > 0

    def test_child_smaller_than_parent(self):
        """child 的 token 数应 ≤ parent_chunk_size。"""
        chunker = ParentChildChunker(
            chunk_size=50, chunk_overlap=5,
            parent_chunk_size=200, parent_chunk_overlap=20,
        )
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert count_tokens(chunk.content) <= 50 * 1.2

    def test_parent_content_is_larger(self):
        """parent_content 应比 child content 更长或相等。"""
        chunker = ParentChildChunker(
            chunk_size=50, chunk_overlap=5,
            parent_chunk_size=200, parent_chunk_overlap=20,
        )
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert len(chunk.metadata["parent_content"]) >= len(chunk.content)

    def test_strategy_tag(self):
        chunker = ParentChildChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk.metadata["chunking_strategy"] == "parent_child"

    def test_global_indices_sequential(self):
        chunker = ParentChildChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(LONG_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ── ChunkerFactory ───────────────────────────────────────────────────


class TestChunkerFactory:
    def test_create_fixed(self):
        c = ChunkerFactory.create("fixed", chunk_size=100, chunk_overlap=10)
        assert isinstance(c, FixedChunker)

    def test_create_recursive(self):
        c = ChunkerFactory.create("recursive", chunk_size=100, chunk_overlap=10)
        assert isinstance(c, RecursiveChunker)

    def test_create_parent_child(self):
        c = ChunkerFactory.create("parent_child", chunk_size=100, chunk_overlap=10)
        assert isinstance(c, ParentChildChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="未知分块策略"):
            ChunkerFactory.create("nonexistent")

    def test_available_strategies(self):
        strategies = ChunkerFactory.available_strategies()
        assert "fixed" in strategies
        assert "recursive" in strategies
        assert "parent_child" in strategies
