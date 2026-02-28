"""HTML 文档解析器"""

from my_rag.domain.models import ParsedDocument
from my_rag.domain.parser.base import BaseParser


class HTMLParser(BaseParser):

    def parse(self, file_path: str) -> ParsedDocument:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        content = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in content.split("\n") if line.strip()]
        content = "\n".join(lines)

        return ParsedDocument(
            content=content,
            metadata={
                "source": file_path,
                "title": title,
                "format": "html",
            },
        )

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]
