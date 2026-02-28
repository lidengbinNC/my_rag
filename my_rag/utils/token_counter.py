"""
Token 计数工具

面试考点：
- 为什么用 Token 而不是字符来衡量分块大小
- tiktoken 编码器的选择（cl100k_base 用于 GPT-3.5/4 系列）
- 惰性加载避免启动开销
"""

import tiktoken

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def warm_up() -> None:
    """预加载 tiktoken 编码器（首次会下载 ~1.7MB 数据，耗时较长）"""
    _get_encoder()


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def encode_tokens(text: str) -> list[int]:
    return _get_encoder().encode(text)


def decode_tokens(tokens: list[int]) -> str:
    return _get_encoder().decode(tokens)
