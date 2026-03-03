"""
通知接口 - 钉钉机器人消息推送

面试考点：
- Webhook 推送模式：调用方 POST 触发，服务端转发到第三方
- 依赖注入：通过 get_dingtalk_notifier() 获取通知器实例
- 服务降级：未配置时返回 503，不影响其他功能
"""

from fastapi import APIRouter, HTTPException

from my_rag.api.schemas.common import APIResponse
from my_rag.api.schemas.notification import (
    DingTalkMarkdownRequest,
    DingTalkSendResult,
    DingTalkTextRequest,
)
from my_rag.core.dependencies import get_dingtalk_notifier
from my_rag.utils.logger import get_logger

router = APIRouter(prefix="/notification")
logger = get_logger(__name__)


def _require_notifier():
    notifier = get_dingtalk_notifier()
    if notifier is None:
        raise HTTPException(status_code=503, detail="钉钉通知未启用，请检查 DINGTALK_ENABLED 配置")
    return notifier


@router.post("/dingtalk/text", response_model=APIResponse[DingTalkSendResult])
async def send_dingtalk_text(body: DingTalkTextRequest):
    """发送钉钉文本消息"""
    notifier = _require_notifier()
    success = await notifier.send_text(content=body.content, at_mobiles=body.at_mobiles)
    return APIResponse(data=DingTalkSendResult(success=success))


@router.post("/dingtalk/markdown", response_model=APIResponse[DingTalkSendResult])
async def send_dingtalk_markdown(body: DingTalkMarkdownRequest):
    """发送钉钉 Markdown 消息"""
    notifier = _require_notifier()
    success = await notifier.send_markdown(title=body.title, text=body.text)
    return APIResponse(data=DingTalkSendResult(success=success))
