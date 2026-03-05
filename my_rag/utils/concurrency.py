"""
并发控制工具

面试考点：
- asyncio.Semaphore：信号量控制并发上限，防止打爆下游服务（API 限流、GPU OOM）
- asyncio.gather：并发执行多个协程，等待全部完成
- 批处理 (Batching)：将 N 个小请求合并为 ceil(N/batch_size) 个批次，减少 I/O 开销
- 对比 Java 的 ThreadPoolExecutor / CompletableFuture
- 生产建议：结合 retry + circuit breaker 做更完善的并发控制
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import TypeVar

from my_rag.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def concurrent_map(
    func: Callable[[T], Coroutine[None, None, R]],
    items: list[T],
    max_concurrency: int = 10,
) -> list[R]:
    """
    并发执行，用 Semaphore 控制最大并发数

    用法：
        results = await concurrent_map(embed_single, texts, max_concurrency=5)
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited(item: T) -> R:
        async with semaphore:
            return await func(item)

    return await asyncio.gather(*[limited(item) for item in items])


async def batch_process(
    func: Callable[[list[T]], Coroutine[None, None, list[R]]],
    items: list[T],
    batch_size: int = 32,
    max_concurrency: int = 3,
) -> list[R]:
    """
    分批 + 并发处理

    将 items 分成多个 batch，每个 batch 调用 func，最多 max_concurrency 个 batch 同时运行。

    用法：
        embeddings = await batch_process(embedding.embed_documents, texts, batch_size=32, max_concurrency=3)
    """
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    logger.info("batch_process_start", total_items=len(items), batches=len(batches), batch_size=batch_size)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_batch(batch: list[T]) -> list[R]:
        async with semaphore:
            return await func(batch)

    batch_results = await asyncio.gather(*[process_batch(b) for b in batches])

    results: list[R] = []
    for batch_result in batch_results:
        results.extend(batch_result)

    return results
