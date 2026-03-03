"""
通知服务抽象基类

面试考点：
- 策略模式：定义通知接口，不同渠道（钉钉、飞书、企业微信）各自实现
- 开闭原则：新增通知渠道只需新增实现类，无需修改调用方
"""

from abc import ABC, abstractmethod


class BaseNotifier(ABC):

    @abstractmethod
    async def send_text(
        self, content: str, at_mobiles: list[str] | None = None,
    ) -> bool:
        """发送纯文本消息"""
        ...

    @abstractmethod
    async def send_markdown(self, title: str, text: str) -> bool:
        """发送 Markdown 富文本消息"""
        ...
