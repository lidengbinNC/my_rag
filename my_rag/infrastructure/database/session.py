"""
数据库会话管理（MySQL + aiomysql 异步驱动）

面试考点：
- SQLAlchemy 2.0 异步引擎（create_async_engine）
- AsyncSession 生命周期管理
- FastAPI 依赖注入 + async generator
- 连接池调优（pool_size / max_overflow / pool_recycle）

MySQL 连接串格式：
  mysql+aiomysql://user:password@host:3306/dbname?charset=utf8mb4

连接池参数说明（面试考点）：
  pool_size       — 连接池常驻连接数（建议 = CPU 核心数 * 2）
  max_overflow    — 超出 pool_size 后允许额外创建的连接数
  pool_timeout    — 等待连接超时秒数
  pool_recycle    — 连接最大存活秒数（MySQL 默认 8h 后断开空闲连接，需小于该值）
  pool_pre_ping   — 每次取连接前发送 SELECT 1 探活，避免使用已断开的连接
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from my_rag.config.settings import settings
from my_rag.infrastructure.database.models import Base

_db_url = settings.database.url

if _db_url.startswith("sqlite"):
    engine = create_async_engine(
        _db_url,
        echo=settings.database.echo,
        future=True,
        connect_args={"check_same_thread": False},
    )
else:
    # MySQL 异步引擎配置
    # pool_recycle=3600 确保连接在 MySQL wait_timeout（默认 8h）之前被回收
    engine = create_async_engine(
        _db_url,
        echo=settings.database.echo,
        future=True,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
        pool_pre_ping=True,          # 自动探活，防止 MySQL 断开空闲连接后报错
    )

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,      # 提交后不过期对象，避免 lazy load 触发额外查询
)


async def init_db() -> None:
    """
    建表（开发 / 首次部署用）

    面试考点：
    - create_all 是幂等操作，表已存在时跳过（不会删除数据）
    - 生产环境应使用 Alembic 做版本化迁移，而非 create_all
    - checkfirst=True（默认）确保不重复建表
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI 依赖注入：提供数据库会话

    面试考点：
    - async generator + context manager 确保 session 在请求结束后被正确关闭
    - 正常结束时 commit，异常时 rollback，与 Spring @Transactional 语义一致
    - expire_on_commit=False 避免 commit 后访问对象属性触发额外 SELECT
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
