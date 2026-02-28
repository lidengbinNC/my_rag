from fastapi import APIRouter
from sqlalchemy import text

from my_rag.api.schemas.common import APIResponse
from my_rag.infrastructure.database.session import async_session_factory

router = APIRouter()


@router.get("/health")
async def health_check() -> APIResponse:
    checks: dict[str, bool] = {}

    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    all_ok = all(checks.values())
    return APIResponse(
        code=200 if all_ok else 503,
        message="healthy" if all_ok else "degraded",
        data={"checks": checks},
    )
