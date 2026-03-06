"""Tests for document parsers and ParserFactory."""

import tempfile
from pathlib import Path

import pytest

from my_rag.domain.models import ParsedDocument
from my_rag.domain.parser.factory import ParserFactory
from my_rag.domain.parser.markdown_parser import MarkdownParser
from my_rag.domain.parser.txt_parser import TxtParser


class TestTxtParser:
    def test_parse_utf8(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("你好世界\nHello World", encoding="utf-8")
        result = TxtParser().parse(str(f))

        assert isinstance(result, ParsedDocument)
        assert "你好世界" in result.content
        assert "Hello World" in result.content
        assert result.metadata["format"] == "txt"

    def test_parse_gbk(self, tmp_path: Path):
        f = tmp_path / "gbk.txt"
        text = "这是一段使用GBK编码的中文文本，用于测试编码检测功能。" * 5
        f.write_bytes(text.encode("gbk"))
        result = TxtParser().parse(str(f))

        assert "GBK" in result.content
        assert "encoding" in result.metadata

    def test_parse_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = TxtParser().parse(str(f))
        assert result.content == ""

    def test_supported_extensions(self):
        assert TxtParser().supported_extensions() == [".txt"]


class TestMarkdownParser:
    def test_parse_with_title(self, tmp_path: Path):
        content = "# My Title\n\nSome content here."
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")

        result = MarkdownParser().parse(str(f))
        assert result.metadata["title"] == "My Title"
        assert result.metadata["format"] == "markdown"
        assert "Some content" in result.content

    def test_parse_without_title(self, tmp_path: Path):
        content = "No heading here.\nJust text."
        f = tmp_path / "no_title.md"
        f.write_text(content, encoding="utf-8")

        result = MarkdownParser().parse(str(f))
        assert result.metadata["title"] == ""

    def test_supported_extensions(self):
        assert MarkdownParser().supported_extensions() == [".md"]


class TestParserFactory:
    def test_get_txt_parser(self):
        parser = ParserFactory.get_parser("document.txt")
        assert isinstance(parser, TxtParser)

    def test_get_md_parser(self):
        parser = ParserFactory.get_parser("readme.md")
        assert isinstance(parser, MarkdownParser)

    def test_case_insensitive(self):
        parser = ParserFactory.get_parser("DOC.TXT")
        assert isinstance(parser, TxtParser)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="不支持的文件类型"):
            ParserFactory.get_parser("data.xyz")

    def test_supported_extensions_list(self):
        exts = ParserFactory.supported_extensions()
        assert ".txt" in exts
        assert ".md" in exts
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".html" in exts
