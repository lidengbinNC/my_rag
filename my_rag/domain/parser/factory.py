"""
解析器工厂

面试考点：
- 工厂模式（Factory Pattern）：根据文件类型自动选择解析器
- 注册表模式：运行时动态注册新解析器，无需修改工厂代码
- 对比 Java 的 @Component + @Autowired 自动注入
"""

from my_rag.domain.parser.base import BaseParser
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ParserFactory:
    _registry: dict[str, BaseParser] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return

        from my_rag.domain.parser.pdf_parser import PDFParser
        from my_rag.domain.parser.docx_parser import DocxParser
        from my_rag.domain.parser.markdown_parser import MarkdownParser
        from my_rag.domain.parser.html_parser import HTMLParser
        from my_rag.domain.parser.txt_parser import TxtParser

        for parser in [PDFParser(), DocxParser(), MarkdownParser(), HTMLParser(), TxtParser()]:
            cls.register(parser)

        cls._initialized = True
        logger.info("parser_factory_initialized", parsers=list(cls._registry.keys()))

    @classmethod
    def register(cls, parser: BaseParser) -> None:
        for ext in parser.supported_extensions():
            cls._registry[ext.lower()] = parser

    @classmethod
    def get_parser(cls, filename: str) -> BaseParser:
        cls._ensure_initialized()

        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()

        parser = cls._registry.get(ext)
        if parser is None:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"不支持的文件类型: '{ext}'，支持: {supported}")

        return parser

    @classmethod
    def supported_extensions(cls) -> list[str]:
        cls._ensure_initialized()
        return sorted(cls._registry.keys())
