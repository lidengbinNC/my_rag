"""
MyRAG 应用入口

面试考点：
- FastAPI 应用生命周期（lifespan）
- 中间件注册顺序
- 静态文件与模板渲染
- CORS 配置
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from my_rag.api.middleware.tracing import RequestTracingMiddleware
from my_rag.api.routes import api_router
from my_rag.config.settings import settings
from my_rag.infrastructure.database.session import init_db
from my_rag.utils.logger import get_logger, setup_logging
from my_rag.utils.token_counter import warm_up as warm_up_tokenizer

BASE_DIR = Path(__file__).resolve().parent

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化资源，关闭时清理"""
    setup_logging(debug=settings.app.debug)
    logger.info("myrag_starting", version=settings.app.version)

    await init_db()
    logger.info("database_initialized")

    settings.upload_path.mkdir(parents=True, exist_ok=True)

    warm_up_tokenizer()
    logger.info("tokenizer_ready")

    yield

    logger.info("myrag_shutdown")


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="智能文档 RAG 问答系统",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(RequestTracingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": settings.app.name,
        "version": settings.app.version,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "my_rag.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
