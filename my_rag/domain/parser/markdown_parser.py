"""Markdown 文档解析器"""

from my_rag.domain.models import ParsedDocument
from my_rag.domain.parser.base import BaseParser


class MarkdownParser(BaseParser):

    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        return ParsedDocument(
            content=content,
            metadata={
                "source": file_path,
                "title": title,
                "format": "markdown",
            },
        )

    def supported_extensions(self) -> list[str]:
        return [".md"]
