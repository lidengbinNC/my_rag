# 面试考点：多阶段构建（Multi-stage Build）减小镜像体积
# Stage 1: 安装依赖
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: 运行时镜像
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY my_rag/ ./my_rag/
COPY .env.example .env

RUN mkdir -p data/uploads data/faiss_index

EXPOSE 8000

CMD ["uvicorn", "my_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
