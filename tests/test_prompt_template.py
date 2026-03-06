"""Tests for prompt template building."""

from my_rag.domain.prompt.template import build_context, build_prompt


class TestBuildContext:
    def test_single_chunk(self):
        chunks = [{"content": "文档内容", "source": "test.txt"}]
        ctx = build_context(chunks)
        assert "文档1" in ctx
        assert "test.txt" in ctx
        assert "文档内容" in ctx

    def test_multiple_chunks(self):
        chunks = [
            {"content": "第一段", "source": "a.txt"},
            {"content": "第二段", "source": "b.txt"},
        ]
        ctx = build_context(chunks)
        assert "文档1" in ctx
        assert "文档2" in ctx
        assert "第一段" in ctx
        assert "第二段" in ctx

    def test_missing_source_defaults(self):
        chunks = [{"content": "内容"}]
        ctx = build_context(chunks)
        assert "未知来源" in ctx

    def test_empty_chunks(self):
        assert build_context([]) == ""


class TestBuildPrompt:
    def test_without_history(self):
        prompt = build_prompt(question="什么是RAG？", context="文档内容")
        assert "什么是RAG？" in prompt
        assert "文档内容" in prompt
        assert "对话历史" not in prompt

    def test_with_history(self):
        prompt = build_prompt(
            question="继续说", context="文档",
            chat_history="用户: 你好\n助手: 你好",
        )
        assert "对话历史" in prompt
        assert "用户: 你好" in prompt
        assert "继续说" in prompt
