"""
PDF 文档解析器

面试考点：
- PyMuPDF (fitz) 文本提取原理
- PDF 解析的三级策略：纯文本 → 表格 → OCR
- 页面级元数据保留
"""

from my_rag.domain.models import ParsedDocument
from my_rag.domain.parser.base import BaseParser
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):

    def parse(self, file_path: str) -> ParsedDocument:
        import fitz

        doc = fitz.open(file_path)
        pages: list[str] = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            if not text.strip():
                text = page.get_text("blocks")
                text = "\n".join(
                    block[4] for block in (text if isinstance(text, list) else [])
                    if isinstance(block, tuple) and len(block) > 4 and isinstance(block[4], str)
                ) if text else ""

            if text.strip():
                pages.append(text.strip())
            else:
                logger.warning("pdf_empty_page", page=page_num, file=file_path)

        doc.close()

        content = "\n\n".join(pages)
        return ParsedDocument(
            content=content,
            metadata={
                "source": file_path,
                "page_count": len(pages),
                "format": "pdf",
            },
            pages=pages,
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
