"""
Span 级链路追踪

面试考点：
- 分布式链路追踪的核心概念：Trace → Span → SpanContext
  - Trace：一次完整请求的调用链
  - Span：Trace 中的一个操作单元（如 embedding、retrieval、llm_call）
  - 每个 Span 记录 name / start_time / duration / attributes
- 对比 OpenTelemetry / Jaeger：本实现是轻量版，用于面试演示
- 实际生产建议用 OpenTelemetry SDK → Jaeger/Zipkin
"""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from my_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    name: str
    trace_id: str = ""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    root_span: Span | None = None
    _span_stack: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes):
        """上下文管理器：自动计时一个操作"""
        s = Span(name=name, trace_id=self.trace_id, attributes=attributes)
        s.start_time = time.perf_counter()

        if self._span_stack:
            self._span_stack[-1].children.append(s)
        elif self.root_span is None:
            self.root_span = s

        self._span_stack.append(s)
        try:
            yield s
        finally:
            s.end_time = time.perf_counter()
            self._span_stack.pop()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "root": self.root_span.to_dict() if self.root_span else None,
        }

    def log_summary(self) -> None:
        """打印链路摘要到日志"""
        if not self.root_span:
            return
        self._log_span(self.root_span, depth=0)

    def _log_span(self, span: Span, depth: int) -> None:
        indent = "  " * depth + ("├── " if depth > 0 else "")
        logger.info(
            "trace_span",
            trace_id=self.trace_id,
            span=f"{indent}[{span.name}] {span.duration_ms}ms",
            **span.attributes,
        )
        for child in span.children:
            self._log_span(child, depth + 1)
