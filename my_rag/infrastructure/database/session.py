"""
数据库会话管理

面试考点：
- SQLAlchemy 2.0 异步引擎
- AsyncSession 生命周期管理
- FastAPI 依赖注入 + async generator
"""

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from my_rag.config.settings import settings
from my_rag.infrastructure.database.models import Base

db_url = settings.database.url
if "sqlite" in db_url:
    db_path = db_url.split("///")[-1]
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    db_url,
    echo=settings.database.echo,
    future=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """建表（开发用，生产环境应使用 Alembic 迁移）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入：提供数据库会话"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
