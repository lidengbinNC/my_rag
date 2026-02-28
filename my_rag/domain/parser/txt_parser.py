"""
纯文本文档解析器

面试考点：
- 文本编码检测（chardet）
- 编码兼容性处理
"""

from my_rag.domain.models import ParsedDocument
from my_rag.domain.parser.base import BaseParser


class TxtParser(BaseParser):

    def parse(self, file_path: str) -> ParsedDocument:
        import chardet

        with open(file_path, "rb") as f:
            raw = f.read()

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        confidence = detected.get("confidence", 0)

        content = raw.decode(encoding, errors="replace")

        return ParsedDocument(
            content=content,
            metadata={
                "source": file_path,
                "encoding": encoding,
                "encoding_confidence": round(confidence, 2),
                "format": "txt",
            },
        )

    def supported_extensions(self) -> list[str]:
        return [".txt"]
