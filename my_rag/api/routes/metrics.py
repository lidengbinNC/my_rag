"""
Prometheus 指标暴露端点

面试考点：
- Prometheus 采集原理：pull 模式，Prometheus 服务器定时 HTTP GET /metrics
- 指标格式：OpenMetrics text format
- 对比 push 模式（Pushgateway）：pull 更适合长期运行的服务
"""

from fastapi import APIRouter
from fastapi.responses import Response

from my_rag.utils.metrics import get_metrics, get_metrics_content_type

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics():
    """暴露 Prometheus 指标（供 Prometheus 服务器抓取）"""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )
