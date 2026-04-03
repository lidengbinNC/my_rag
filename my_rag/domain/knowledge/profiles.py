"""知识库画像相关的纯函数工具。"""

from __future__ import annotations

import json


def loads_str_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def profile_matches(
    *,
    profile_domain: str,
    profile_visibility: str,
    profile_language: str,
    tags_json: str | None,
    allowed_roles_json: str | None,
    domain: str = "",
    visibility: str = "",
    language: str = "",
    tag: str = "",
    role: str = "",
) -> bool:
    if domain and profile_domain != domain:
        return False
    if visibility and profile_visibility != visibility:
        return False
    if language and profile_language != language:
        return False
    tags = loads_str_list(tags_json)
    allowed_roles = loads_str_list(allowed_roles_json)
    if tag and tag not in tags:
        return False
    if role and allowed_roles and role not in allowed_roles:
        return False
    return True
