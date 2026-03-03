"""
通知相关 Schema
"""

from pydantic import BaseModel, Field


class DingTalkTextRequest(BaseModel):
    """钉钉文本消息请求"""
    content: str = Field(..., min_length=1, description="消息内容")
    at_mobiles: list[str] | None = Field(default=None, description="被@人的手机号列表")


class DingTalkMarkdownRequest(BaseModel):
    """钉钉 Markdown 消息请求"""
    title: str = Field(..., min_length=1, description="消息标题（会话列表中展示）")
    text: str = Field(..., min_length=1, description="Markdown 格式消息正文")


class DingTalkSendResult(BaseModel):
    """钉钉发送结果"""
    success: bool = Field(description="是否发送成功")
