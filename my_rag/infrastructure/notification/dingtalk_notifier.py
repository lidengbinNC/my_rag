"""
钉钉自定义机器人通知实现

面试考点：
- Webhook 模式：通过 HTTP POST 推送消息，无需长连接
- 加签安全：HmacSHA256 签名防止 Webhook 地址泄露后被滥用
- 异步 HTTP 客户端：httpx.AsyncClient 配合 asyncio 实现非阻塞推送
- 消息类型：text / markdown / link / actionCard / feedCard
"""

import base64
import hashlib
import hmac
import time
import urllib.parse

import httpx

from my_rag.infrastructure.notification.base import BaseNotifier
from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


class DingTalkNotifier(BaseNotifier):
    """钉钉群自定义机器人"""

    def __init__(self, webhook_url: str, secret: str = ""):
        self._webhook_url = webhook_url
        self._secret = secret

    def _build_signed_url(self) -> str:
        """构造带签名的请求 URL（加签安全模式）"""
        if not self._secret:
            return self._webhook_url

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self._secret}"
        hmac_code = hmac.new(
            self._secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return f"{self._webhook_url}&timestamp={timestamp}&sign={sign}"

    async def send_text(
        self, content: str, at_mobiles: list[str] | None = None,
    ) -> bool:
        payload = {
            "msgtype": "text",
            "text": {"content": content},
            "at": {
                "atMobiles": at_mobiles or [],
                "isAtAll": False,
            },
        }
        return await self._post(payload)

    async def send_markdown(self, title: str, text: str) -> bool:
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": text},
        }
        return await self._post(payload)

    async def send_link(
        self, title: str, text: str, message_url: str, pic_url: str = "",
    ) -> bool:
        """发送链接类型消息"""
        payload = {
            "msgtype": "link",
            "link": {
                "title": title,
                "text": text,
                "messageUrl": message_url,
                "picUrl": pic_url,
            },
        }
        return await self._post(payload)

    async def _post(self, payload: dict) -> bool:
        url = self._build_signed_url()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, json=payload, timeout=10,
                    headers={"Content-Type": "application/json"},
                )
                result = resp.json()

            errcode = result.get("errcode", -1)
            if errcode != 0:
                logger.error(
                    "dingtalk_send_failed",
                    errcode=errcode,
                    errmsg=result.get("errmsg"),
                )
                return False

            logger.info("dingtalk_send_ok", msgtype=payload.get("msgtype"))
            return True

        except Exception as exc:
            logger.error("dingtalk_send_error", error=str(exc))
            return False
