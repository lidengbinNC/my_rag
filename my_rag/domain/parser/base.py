"""
文档解析器抽象基类

面试考点：
- 抽象基类（ABC）定义接口规范（对比 Java 的 interface）
- 策略模式：不同文件类型使用不同解析策略
"""

from abc import ABC, abstractmethod

from my_rag.domain.models import ParsedDocument


class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文件，返回结构化的文档内容"""
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """返回该解析器支持的文件扩展名列表"""
        ...
