"""
请求追踪 + Prometheus 指标中间件

面试考点：
- FastAPI 中间件机制（类比 Java 的 Filter / Interceptor）
- 请求 ID 追踪（分布式链路追踪基础）
- 请求耗时统计 → Histogram 分位数（p50/p95/p99）
- structlog contextvars：在整个请求生命周期中自动携带 request_id
"""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from my_rag.utils.logger import get_logger
from my_rag.utils.metrics import REQUEST_COUNT, REQUEST_DURATION

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        duration_ms = round(duration * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        if not endpoint.startswith("/static"):
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

            logger.info(
                "request_completed",
                method=method,
                path=endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        return response
