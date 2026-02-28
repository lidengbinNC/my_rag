# MyRAG - 智能文档问答系统 设计文档

## 一、项目概述

### 1.1 项目定位

MyRAG 是一个基于 RAG（Retrieval-Augmented Generation）架构的智能文档问答系统。支持多格式文档上传、智能解析、语义检索和大模型问答，提供完整的从文档摄入到智能对话的端到端解决方案。

### 1.2 技术栈总览

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步高性能 API 框架 |
| 任务队列 | Celery + Redis | 异步文档处理任务 |
| 向量数据库 | Milvus (主) / FAISS (轻量) | 可切换的向量存储后端 |
| 关系数据库 | PostgreSQL + SQLAlchemy | 元数据与用户管理 |
| 缓存 | Redis | 查询缓存、会话管理 |
| Embedding | BGE-M3 (本地) / OpenAI API | 多模型支持 |
| LLM | Qwen / DeepSeek / OpenAI | 可插拔 LLM 后端 |
| Reranker | BGE-Reranker / Cohere | 二阶段检索重排 |
| 前端 | Gradio / Streamlit | 快速原型 UI |
| 容器化 | Docker + Docker Compose | 一键部署 |
| 监控 | Prometheus + Grafana | 系统可观测性 |

### 1.3 面试知识点覆盖矩阵

> 本项目的模块设计确保覆盖以下 AI 系统工程师面试高频考点：

| 知识领域 | 考点 | 对应模块 |
|----------|------|---------|
| RAG 架构 | Naive RAG / Advanced RAG / Modular RAG | 整体架构 |
| 文档处理 | PDF解析、OCR、表格提取 | DocumentParser |
| 分块策略 | Fixed / Recursive / Semantic Chunking | ChunkingEngine |
| Embedding | Dense / Sparse / Multi-Vector | EmbeddingService |
| 向量检索 | ANN 算法、HNSW、IVF | VectorStore |
| 混合检索 | Dense + Sparse + RRF 融合 | HybridRetriever |
| 重排序 | Cross-Encoder Reranking | Reranker |
| Prompt 工程 | 模板管理、Few-Shot、CoT | PromptEngine |
| LLM 集成 | 流式输出、多模型切换、Token 管理 | LLMService |
| 对话管理 | 多轮对话、历史压缩、上下文窗口 | ConversationManager |
| 评估体系 | Faithfulness / Relevance / Recall | EvaluationFramework |
| 异步编程 | async/await、并发控制 | 全局 |
| API 设计 | RESTful、WebSocket、中间件 | API Layer |
| 缓存策略 | 语义缓存、多级缓存 | CacheManager |
| 系统设计 | 高可用、可扩展、解耦 | 整体架构 |
| DevOps | Docker、CI/CD、健康检查 | 部署模块 |
| 可观测性 | Logging / Metrics / Tracing | Observability |
| Python 工程 | 设计模式、类型注解、包管理 | 全局 |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Gradio  │  │ REST API │  │WebSocket │  │  SDK (Python)    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────────┘   │
└───────┼──────────────┼──────────────┼───────────────┼───────────────┘
        │              │              │               │
┌───────▼──────────────▼──────────────▼───────────────▼───────────────┐
│                        API Gateway Layer                             │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────┐      │
│  │ Auth (JWT) │ │RateLimit │ │  CORS     │ │ RequestTracing │      │
│  └────────────┘ └──────────┘ └───────────┘ └────────────────┘      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       Application Layer                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   RAG Pipeline Orchestrator                  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │  Query   │ │Retrieval │ │ Rerank   │ │  Generation  │  │    │
│  │  │ Process  │→│  Engine  │→│  Engine  │→│   Engine     │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐    │
│  │  Document Pipeline   │  │     Conversation Manager         │    │
│  │  ┌──────┐ ┌───────┐  │  │  ┌─────────┐ ┌──────────────┐  │    │
│  │  │Parse │→│ Chunk │  │  │  │ History │ │ Context      │  │    │
│  │  └──────┘ └───┬───┘  │  │  │ Manager │ │ Compressor   │  │    │
│  │        ┌──────▼─────┐ │  │  └─────────┘ └──────────────┘  │    │
│  │        │  Embed     │ │  └──────────────────────────────────┘    │
│  │        └──────┬─────┘ │                                          │
│  │        ┌──────▼─────┐ │                                          │
│  │        │  Index     │ │                                          │
│  │        └────────────┘ │                                          │
│  └──────────────────────┘                                           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       Infrastructure Layer                           │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │  Milvus  │ │PostgreSQL│ │  Redis   │ │  Celery  │ │ MinIO   │  │
│  │(Vectors) │ │(Metadata)│ │ (Cache)  │ │ (Tasks)  │ │(Storage)│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └─────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Observability Stack                              │   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────────────┐    │   │
│  │  │ Prometheus │  │ Grafana  │  │  Structured Logging   │    │   │
│  │  └────────────┘  └──────────┘  └───────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 分层架构说明

项目采用 **Clean Architecture（整洁架构）** 思想，共分四层：

1. **接口层（API Layer）**：处理 HTTP/WebSocket 请求，参数校验，响应序列化
2. **应用层（Application Layer）**：编排业务流程，RAG Pipeline 调度
3. **领域层（Domain Layer）**：核心业务逻辑，文档处理、检索、生成等
4. **基础设施层（Infrastructure Layer）**：数据库、缓存、消息队列等外部依赖

> **面试考点**：分层架构、依赖倒置原则（DIP）、关注点分离。作为 Java 转型者，展示你理解架构设计并能用 Python 实现。

---

## 三、项目目录结构

```
my_rag/
├── docs/                          # 项目文档
│   ├── DESIGN.md                  # 本设计文档
│   └── API.md                     # API 接口文档
│
├── src/
│   └── my_rag/                    # 主包
│       ├── __init__.py
│       ├── main.py                # FastAPI 应用入口
│       ├── config/                # 配置管理
│       │   ├── __init__.py
│       │   └── settings.py        # Pydantic Settings，多环境配置
│       │
│       ├── api/                   # 接口层
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── document.py    # 文档上传/管理接口
│       │   │   ├── chat.py        # 对话接口（含 WebSocket 流式）
│       │   │   ├── knowledge.py   # 知识库管理接口
│       │   │   └── evaluation.py  # 评估接口
│       │   ├── middleware/
│       │   │   ├── auth.py        # JWT 认证中间件
│       │   │   ├── rate_limit.py  # 限流中间件
│       │   │   └── tracing.py     # 请求追踪中间件
│       │   └── schemas/           # Pydantic 请求/响应模型
│       │       ├── document.py
│       │       ├── chat.py
│       │       └── common.py
│       │
│       ├── core/                  # 应用层 - 业务编排
│       │   ├── __init__.py
│       │   ├── rag_pipeline.py    # RAG Pipeline 编排器
│       │   ├── document_pipeline.py  # 文档处理流水线
│       │   └── conversation.py    # 多轮对话管理
│       │
│       ├── domain/                # 领域层 - 核心业务
│       │   ├── __init__.py
│       │   ├── parser/            # 文档解析器
│       │   │   ├── base.py        # 解析器抽象基类
│       │   │   ├── pdf_parser.py  # PDF 解析（含 OCR）
│       │   │   ├── docx_parser.py # Word 文档解析
│       │   │   ├── markdown_parser.py
│       │   │   ├── html_parser.py
│       │   │   └── factory.py     # 解析器工厂（工厂模式）
│       │   │
│       │   ├── chunking/          # 文本分块
│       │   │   ├── base.py        # 分块策略抽象基类
│       │   │   ├── fixed_chunker.py      # 固定大小分块
│       │   │   ├── recursive_chunker.py  # 递归字符分块
│       │   │   ├── semantic_chunker.py   # 语义分块
│       │   │   └── factory.py     # 分块策略工厂
│       │   │
│       │   ├── embedding/         # 向量化
│       │   │   ├── base.py        # Embedding 抽象基类
│       │   │   ├── local_embedding.py    # 本地模型（BGE-M3）
│       │   │   ├── openai_embedding.py   # OpenAI API
│       │   │   └── factory.py
│       │   │
│       │   ├── retrieval/         # 检索引擎
│       │   │   ├── base.py        # 检索器抽象基类
│       │   │   ├── dense_retriever.py    # 稠密向量检索
│       │   │   ├── sparse_retriever.py   # 稀疏检索（BM25）
│       │   │   ├── hybrid_retriever.py   # 混合检索 + RRF
│       │   │   └── query_transform.py    # 查询改写/扩展
│       │   │
│       │   ├── reranker/          # 重排序
│       │   │   ├── base.py
│       │   │   ├── cross_encoder_reranker.py  # Cross-Encoder
│       │   │   └── llm_reranker.py            # LLM-based Reranker
│       │   │
│       │   ├── llm/               # LLM 服务
│       │   │   ├── base.py        # LLM 抽象基类
│       │   │   ├── openai_llm.py  # OpenAI 兼容接口
│       │   │   ├── local_llm.py   # 本地模型（Ollama）
│       │   │   └── factory.py
│       │   │
│       │   └── prompt/            # Prompt 工程
│       │       ├── template.py    # Prompt 模板管理
│       │       ├── few_shot.py    # Few-Shot 示例管理
│       │       └── templates/     # 模板文件目录
│       │           ├── qa.py      # 问答 Prompt
│       │           ├── summary.py # 摘要 Prompt
│       │           └── judge.py   # 评估 Prompt
│       │
│       ├── infrastructure/        # 基础设施层
│       │   ├── __init__.py
│       │   ├── database/
│       │   │   ├── models.py      # SQLAlchemy ORM 模型
│       │   │   ├── repository.py  # Repository 模式
│       │   │   └── session.py     # 数据库连接管理
│       │   ├── vector_store/
│       │   │   ├── base.py        # 向量库抽象基类
│       │   │   ├── milvus_store.py
│       │   │   └── faiss_store.py
│       │   ├── cache/
│       │   │   ├── base.py
│       │   │   ├── redis_cache.py
│       │   │   └── semantic_cache.py  # 语义缓存
│       │   ├── storage/
│       │   │   ├── base.py        # 文件存储抽象基类
│       │   │   ├── local_storage.py
│       │   │   └── minio_storage.py
│       │   └── celery/
│       │       ├── app.py         # Celery 应用配置
│       │       └── tasks.py       # 异步任务定义
│       │
│       ├── evaluation/            # 评估框架
│       │   ├── __init__.py
│       │   ├── metrics.py         # 评估指标（RAGAS 风格）
│       │   ├── dataset.py         # 评估数据集管理
│       │   └── evaluator.py       # 评估器
│       │
│       └── utils/                 # 工具模块
│           ├── __init__.py
│           ├── logger.py          # 结构化日志
│           ├── metrics.py         # Prometheus 指标
│           └── token_counter.py   # Token 计数器
│
├── tests/                         # 测试
│   ├── unit/                      # 单元测试
│   ├── integration/               # 集成测试
│   └── e2e/                       # 端到端测试
│
├── scripts/                       # 运维脚本
│   ├── init_db.py                 # 数据库初始化
│   └── benchmark.py               # 性能基准测试
│
├── docker/                        # Docker 配置
│   ├── Dockerfile
│   ├── Dockerfile.worker          # Celery Worker 镜像
│   └── docker-compose.yml         # 一键启动全部服务
│
├── pyproject.toml                 # 项目元数据 & 依赖 (PEP 621)
├── .env.example                   # 环境变量示例
├── .gitignore
└── Makefile                       # 常用命令快捷入口
```

---

## 四、核心模块详细设计

### 4.1 文档处理流水线（Document Pipeline）

#### 4.1.1 文档解析（Parser）

> **面试考点**：工厂模式、策略模式、多格式文件处理、OCR 技术

```
支持格式：PDF / DOCX / Markdown / HTML / TXT
          │
          ▼
    ┌─────────────┐
    │ ParserFactory│ ── 根据 MIME Type / 后缀名选择解析器
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ BaseParser  │ (抽象基类, ABC)
    ├─────────────┤
    │ + parse()   │ → List[Document]
    │ + supports()│ → bool
    └──────┬──────┘
           │
    ┌──────┼──────────────┬────────────────┐
    ▼      ▼              ▼                ▼
  PDF    DOCX         Markdown           HTML
  Parser  Parser       Parser            Parser
    │
    ├── PyMuPDF (文本提取)
    ├── pdfplumber (表格提取)
    └── PaddleOCR / Tesseract (扫描件 OCR)
```

**关键设计**：
- 使用 **工厂模式（Factory Pattern）** 根据文件类型自动选择解析器
- 使用 **策略模式（Strategy Pattern）** 使解析策略可插拔
- PDF 解析分三级：纯文本提取 → 表格提取 → OCR 兜底
- 解析结果统一为 `Document` 数据模型（包含 content、metadata、source 等字段）

#### 4.1.2 文本分块（Chunking）

> **面试考点**：分块策略对检索质量的影响、Chunk Size 与 Overlap 调优、语义分块原理

```python
# 核心抽象
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        ...
```

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| FixedChunker | 按固定 Token 数分块，带重叠窗口 | 通用文本，基线方案 |
| RecursiveChunker | 按分隔符层级递归分割（段落→句子→词） | 结构化文本 |
| SemanticChunker | 基于 Embedding 相似度断点分块 | 需要语义完整性的场景 |

**关键参数**：
- `chunk_size`：块大小（默认 512 tokens）
- `chunk_overlap`：重叠大小（默认 50 tokens）
- `separators`：递归分割符层级

**Parent-Child 分块策略**：
- 存储时保留 **大块（Parent）** 和 **小块（Child）** 的映射关系
- 检索时用小块匹配，返回时用大块保证上下文完整
- 这是 Advanced RAG 的核心技巧之一

#### 4.1.3 向量化（Embedding）

> **面试考点**：Embedding 模型选型、向量维度、批处理优化、多语言支持

```python
class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """查询向量化（可能使用不同 instruction）"""
        ...
```

**模型选型**：
| 模型 | 维度 | 特点 |
|------|------|------|
| BGE-M3 | 1024 | 支持 Dense + Sparse + ColBERT 多向量 |
| text-embedding-3-small | 1536 | OpenAI，高质量，按 API 计费 |
| bge-large-zh-v1.5 | 1024 | 中文优化 |

**优化策略**：
- **批处理**：控制 batch_size 避免 OOM
- **异步并发**：使用 `asyncio.Semaphore` 控制并发数
- **缓存**：相同文本的 Embedding 结果缓存到 Redis
- **量化**：支持 float16 降低存储开销

---

### 4.2 检索引擎（Retrieval Engine）

> **面试考点**：这是 RAG 面试的核心考区

#### 4.2.1 检索架构

```
用户查询
    │
    ▼
┌──────────────────┐
│  Query Transform │ ── 查询理解与改写
│  ├─ 查询改写     │    (HyDE / Multi-Query / Step-Back)
│  ├─ 查询扩展     │
│  └─ 意图识别     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│           Hybrid Retrieval               │
│                                          │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │Dense Retriever│  │Sparse Retriever │  │
│  │ (Vector ANN)  │  │   (BM25)        │  │
│  │               │  │                  │  │
│  │ HNSW / IVF   │  │ Elasticsearch /  │  │
│  │              │  │ rank_bm25       │  │
│  └──────┬───────┘  └────────┬─────────┘  │
│         │                   │            │
│         └────────┬──────────┘            │
│                  ▼                       │
│         ┌───────────────┐                │
│         │  RRF Fusion   │  ← 倒数排名融合 │
│         └───────┬───────┘                │
└─────────────────┼────────────────────────┘
                  │
                  ▼
         ┌───────────────┐
         │   Reranker    │ ── Cross-Encoder 精排
         └───────┬───────┘
                 │
                 ▼
          Top-K Documents
```

#### 4.2.2 稠密检索（Dense Retrieval）

> **面试考点**：ANN 算法、HNSW 原理、IVF 索引、向量相似度度量

- **向量相似度**：支持 Cosine Similarity / Inner Product / L2 Distance
- **ANN 索引**：
  - **HNSW**（Hierarchical Navigable Small World）：高召回、适合中小规模
  - **IVF_FLAT** / **IVF_PQ**：适合大规模数据，支持量化压缩
- **元数据过滤**：支持按文档来源、时间范围、标签等进行预过滤

#### 4.2.3 稀疏检索（Sparse Retrieval - BM25）

> **面试考点**：BM25 公式、TF-IDF 原理、倒排索引

- 使用 `rank_bm25` 库或 Elasticsearch 实现
- 对关键词、专业术语、ID 类查询有独特优势
- 与稠密检索形成互补

#### 4.2.4 混合检索与 RRF 融合

> **面试考点**：Reciprocal Rank Fusion 原理、混合检索为什么优于单一检索

```python
# RRF 核心公式
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# 融合逻辑：对每个候选文档，将其在各检索器中的 RRF 分数相加
```

**配置参数**：
- `dense_weight`：稠密检索权重
- `sparse_weight`：稀疏检索权重
- `rrf_k`：RRF 常数（通常为 60）
- `top_k`：最终返回文档数

#### 4.2.5 查询改写（Query Transform）

> **面试考点**：HyDE、Multi-Query、Step-Back Prompting

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| HyDE | 让 LLM 生成假设性答案，用答案做检索 | 提升语义匹配质量 |
| Multi-Query | 将原始查询改写为多个不同角度的查询 | 提升召回率 |
| Step-Back | 先问更高层次的问题，再回答具体问题 | 复杂推理问题 |
| Query Decomposition | 将复杂问题分解为子问题 | 多跳推理 |

#### 4.2.6 重排序（Reranking）

> **面试考点**：Bi-Encoder vs Cross-Encoder、重排序为什么能提升精度

```
Bi-Encoder (Retrieval)         Cross-Encoder (Reranking)
┌──────┐  ┌──────┐             ┌─────────────────┐
│Query │  │ Doc  │             │ [CLS] Query      │
│      │  │      │             │ [SEP] Document   │
│Encode│  │Encode│             │ [SEP]            │
└──┬───┘  └──┬───┘             └────────┬─────────┘
   │         │                          │
   ▼         ▼                          ▼
 cosine similarity               relevance score
 (快速，适合召回)               (精准，适合精排)
```

- **阶段一**：Bi-Encoder 快速召回 Top-100
- **阶段二**：Cross-Encoder 对 Top-100 精确打分，返回 Top-5

---

### 4.3 生成引擎（Generation Engine）

#### 4.3.1 LLM 服务

> **面试考点**：LLM API 集成、流式输出、Token 管理、错误处理

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """同步生成"""
        ...

    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成（SSE）"""
        ...
```

**关键设计**：
- 统一的 LLM 抽象接口，支持 OpenAI / Ollama / vLLM 等后端
- **流式输出**：通过 SSE（Server-Sent Events）实现逐字输出
- **Token 管理**：
  - 使用 `tiktoken` 精确计算 Token 数
  - 动态调整 Context Window：System Prompt + History + Retrieved Docs + Query ≤ Max Tokens
  - 当历史记录过长时，自动触发对话压缩
- **重试与降级**：指数退避重试、模型降级策略
- **并发控制**：使用信号量限制对 LLM API 的并发请求数

#### 4.3.2 Prompt 工程

> **面试考点**：Prompt 模板设计、Few-Shot、CoT、Prompt 版本管理

```python
# 核心 QA Prompt 模板结构
QA_PROMPT = """
你是一个专业的文档问答助手。请严格根据以下参考文档回答用户问题。

## 规则
1. 只根据参考文档中的内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确告知用户
3. 回答时引用文档来源

## 参考文档
{context}

## 对话历史
{chat_history}

## 用户问题
{question}

## 回答
"""
```

**Prompt 管理策略**：
- 使用 Jinja2 模板引擎管理复杂 Prompt
- 支持 Prompt 版本控制和 A/B 测试
- Few-Shot 示例动态选择（基于查询相似度）

#### 4.3.3 多轮对话管理

> **面试考点**：对话历史管理、上下文窗口管理、对话压缩

```
用户 Query → 是否包含指代/省略？
                │
         ┌──────┼───────┐
         │ Yes          │ No
         ▼              ▼
  查询改写(融入历史)    直接检索
         │              │
         └──────┬───────┘
                ▼
         正常 RAG 流程
```

**对话历史策略**：
| 策略 | 说明 |
|------|------|
| Sliding Window | 保留最近 N 轮对话 |
| Token Budget | 按 Token 预算裁剪历史 |
| Summary | LLM 压缩历史为摘要 |
| Hybrid | 近期完整保留 + 远期摘要 |

---

### 4.4 知识库管理

> **面试考点**：多租户设计、Collection 管理、增量更新

```
┌─────────────────────────────────────────┐
│              Knowledge Base             │
│                                         │
│  ┌───────────────┐  ┌───────────────┐  │
│  │  Collection A │  │ Collection B  │  │
│  │  (技术文档)    │  │  (产品手册)   │  │
│  │               │  │               │  │
│  │  Chunks: 1200 │  │  Chunks: 800  │  │
│  │  Docs: 45     │  │  Docs: 23     │  │
│  └───────────────┘  └───────────────┘  │
│                                         │
│  Metadata: PostgreSQL                   │
│  Vectors:  Milvus                       │
│  Files:    MinIO / Local                │
└─────────────────────────────────────────┘
```

**核心功能**：
- 创建/删除知识库（对应 Milvus Collection）
- 文档增删改查
- 增量索引：仅处理新增/变更文档
- 文档去重：基于内容哈希 + 语义去重

---

### 4.5 缓存体系

> **面试考点**：缓存策略、语义缓存、缓存一致性

```
┌────────────────────────────────────────────────┐
│                  Cache Layers                   │
│                                                 │
│  L1: Embedding Cache (Redis)                    │
│      Key: hash(text) → embedding vector         │
│      TTL: 7 days                                │
│                                                 │
│  L2: Semantic Cache (Redis + Vector)            │
│      Key: query_embedding → cached_response     │
│      Threshold: cosine_sim > 0.95               │
│      TTL: 1 hour                                │
│                                                 │
│  L3: Result Cache (Redis)                       │
│      Key: hash(query + kb_id + params)          │
│      TTL: 5 minutes                             │
│                                                 │
└────────────────────────────────────────────────┘
```

**语义缓存原理**：
- 对用户 Query 计算 Embedding
- 在缓存向量库中检索最相似的历史 Query
- 如果相似度 > 阈值（如 0.95），直接返回缓存结果
- 否则执行完整 RAG 流程并缓存结果

---

## 五、API 设计

### 5.1 RESTful API

> **面试考点**：RESTful 设计规范、状态码使用、分页设计、错误处理

```
# 知识库管理
POST   /api/v1/knowledge-bases              # 创建知识库
GET    /api/v1/knowledge-bases              # 列表（分页）
GET    /api/v1/knowledge-bases/{id}         # 详情
DELETE /api/v1/knowledge-bases/{id}         # 删除

# 文档管理
POST   /api/v1/knowledge-bases/{id}/documents          # 上传文档
GET    /api/v1/knowledge-bases/{id}/documents          # 文档列表
DELETE /api/v1/knowledge-bases/{id}/documents/{doc_id} # 删除文档
GET    /api/v1/documents/{doc_id}/chunks               # 查看分块

# 对话
POST   /api/v1/chat                         # 单轮问答
POST   /api/v1/chat/stream                  # 流式问答 (SSE)
WS     /api/v1/chat/ws                      # WebSocket 对话

# 对话历史
GET    /api/v1/conversations                # 对话列表
GET    /api/v1/conversations/{id}/messages  # 对话消息
DELETE /api/v1/conversations/{id}           # 删除对话

# 评估
POST   /api/v1/evaluations                  # 触发评估
GET    /api/v1/evaluations/{id}             # 评估结果

# 系统
GET    /api/v1/health                       # 健康检查
GET    /api/v1/metrics                      # Prometheus 指标
```

### 5.2 统一响应格式

```json
{
    "code": 200,
    "message": "success",
    "data": { ... },
    "request_id": "uuid-xxx",
    "timestamp": "2026-02-27T10:00:00Z"
}
```

### 5.3 流式响应（SSE）

```
POST /api/v1/chat/stream

Response (text/event-stream):
data: {"type": "retrieval", "documents": [...]}
data: {"type": "token", "content": "根据"}
data: {"type": "token", "content": "文档"}
data: {"type": "token", "content": "记载"}
...
data: {"type": "sources", "references": [...]}
data: {"type": "done", "usage": {"prompt_tokens": 1200, "completion_tokens": 300}}
```

---

## 六、Advanced RAG 策略

> **面试考点**：这些是区分候选人深度的关键知识点

### 6.1 Self-RAG（自反思 RAG）

```
Query → Retrieval → 是否需要检索？ (LLM 判断)
                         │
                  ┌──────┼──────┐
                  │ Yes         │ No
                  ▼             ▼
             检索文档        直接生成
                  │
                  ▼
         生成回答 + 自我评估
                  │
         ┌───────┼────────┐
         │ 通过           │ 不通过
         ▼               ▼
       返回结果      重新检索/生成
```

### 6.2 Corrective RAG（纠正式 RAG）

```
检索文档 → 相关性评估（对每篇文档打分）
              │
    ┌─────────┼──────────┐
    │ Correct │ Ambiguous │ Incorrect
    ▼         ▼           ▼
  使用      补充检索     Web 搜索兜底
```

### 6.3 RAG Fusion

```
原始 Query → 生成 N 个变体查询
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
      Query1  Query2  Query3
        │       │       │
        ▼       ▼       ▼
      检索1   检索2   检索3
        │       │       │
        └───────┼───────┘
                ▼
          RRF 融合排序
                │
                ▼
          Top-K 结果
```

### 6.4 Graph RAG（可选进阶）

- 从文档中抽取实体和关系，构建知识图谱
- 将图结构与向量检索结合
- 适合需要多跳推理的复杂问题

---

## 七、评估体系

> **面试考点**：RAG 评估指标、评估自动化、A/B 测试

### 7.1 评估指标（RAGAS 风格）

| 指标 | 评估目标 | 计算方式 |
|------|---------|---------|
| **Faithfulness** | 回答是否忠实于检索文档 | LLM 判断每句话是否有文档支撑 |
| **Answer Relevancy** | 回答是否与问题相关 | 从回答反向生成问题，计算与原问题的相似度 |
| **Context Precision** | 检索文档排序质量 | 相关文档是否排在前面 |
| **Context Recall** | 检索文档覆盖度 | Ground Truth 中有多少信息被检索到 |
| **Answer Correctness** | 回答的正确性 | 与 Ground Truth 的语义相似度 + 事实重叠 |

### 7.2 评估流程

```
评估数据集 (Question + Ground Truth + Context)
                    │
                    ▼
            ┌───────────────┐
            │ RAG Pipeline  │ → Generated Answer
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  Evaluator    │
            │               │
            │ • Faithfulness│
            │ • Relevancy   │
            │ • Precision   │
            │ • Recall      │
            └───────┬───────┘
                    │
                    ▼
            评估报告 (JSON + 可视化)
```

### 7.3 持续评估

- 每次配置变更（分块参数、Embedding 模型、检索策略等）后自动运行评估
- 结果存储到数据库，支持纵向对比
- 支持 A/B 测试不同 RAG 配置

---

## 八、可观测性设计

> **面试考点**：Logging / Metrics / Tracing 三大支柱

### 8.1 结构化日志

```python
# 使用 structlog，输出 JSON 格式
{
    "timestamp": "2026-02-27T10:00:00Z",
    "level": "info",
    "event": "rag_pipeline_completed",
    "request_id": "uuid-xxx",
    "user_id": "user-123",
    "query": "什么是向量数据库？",
    "retrieval_count": 5,
    "rerank_count": 3,
    "latency_ms": 1200,
    "token_usage": {"prompt": 800, "completion": 200}
}
```

### 8.2 指标监控（Prometheus）

```
# 核心指标
rag_request_total                   # 总请求数
rag_request_duration_seconds        # 请求耗时分布
rag_retrieval_documents_total       # 检索文档数
rag_llm_token_usage_total           # Token 使用量
rag_cache_hit_ratio                 # 缓存命中率
rag_document_processing_duration    # 文档处理耗时
rag_embedding_batch_size            # Embedding 批次大小
```

### 8.3 链路追踪

```
[Request] ─── trace_id: abc123
    │
    ├── [QueryTransform] 12ms
    │
    ├── [DenseRetrieval] 45ms
    │     └── [MilvusSearch] 30ms
    │
    ├── [SparseRetrieval] 20ms
    │     └── [BM25Search] 15ms
    │
    ├── [RRFFusion] 5ms
    │
    ├── [Reranking] 80ms
    │     └── [CrossEncoder] 75ms
    │
    └── [Generation] 800ms
          ├── [PromptBuild] 5ms
          └── [LLMCall] 795ms
```

---

## 九、Python 工程化实践

> **面试考点**：作为 Java 转型者，展示你掌握了 Python 的工程化最佳实践

### 9.1 设计模式应用

| 模式 | 应用场景 | 实现方式 |
|------|---------|---------|
| 工厂模式 | Parser / Chunker / LLM 创建 | Factory 类 + 注册表 |
| 策略模式 | 检索策略、分块策略可切换 | ABC + 配置驱动 |
| 观察者模式 | Pipeline 事件通知 | 回调函数 / Event Emitter |
| 责任链模式 | 中间件链 | FastAPI Middleware |
| 单例模式 | 数据库连接、模型加载 | `@lru_cache` / 模块级实例 |
| Repository 模式 | 数据访问 | SQLAlchemy Repository |
| Pipeline 模式 | RAG 流程编排 | 可组合的 Pipeline 组件 |

### 9.2 类型注解

```python
from typing import Protocol, AsyncIterator

class Retriever(Protocol):
    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        ...
```

### 9.3 异步编程

```python
# 并发控制示例
semaphore = asyncio.Semaphore(10)

async def embed_with_limit(text: str) -> list[float]:
    async with semaphore:
        return await embedding_model.embed(text)

# 批量并发
results = await asyncio.gather(
    *[embed_with_limit(t) for t in texts]
)
```

### 9.4 依赖注入

```python
# 使用 FastAPI 的依赖注入系统
def get_rag_pipeline(
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm: BaseLLM = Depends(get_llm),
) -> RAGPipeline:
    return RAGPipeline(retriever, reranker, llm)

@router.post("/chat")
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    ...
```

### 9.5 错误处理

```python
# 自定义异常层级
class MyRAGException(Exception): ...
class DocumentParseError(MyRAGException): ...
class EmbeddingError(MyRAGException): ...
class RetrievalError(MyRAGException): ...
class LLMError(MyRAGException): ...
class RateLimitError(LLMError): ...

# 全局异常处理器
@app.exception_handler(MyRAGException)
async def handle_myrag_error(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.error_code, "message": str(exc)}
    )
```

---

## 十、部署方案

### 10.1 Docker Compose 架构

```yaml
services:
  api:          # FastAPI 主服务
  worker:       # Celery Worker（文档处理）
  postgres:     # 元数据存储
  redis:        # 缓存 + 消息队列
  milvus:       # 向量数据库
  minio:        # 文件存储
  prometheus:   # 指标采集
  grafana:      # 可视化监控面板
```

### 10.2 环境配置

```bash
# .env 配置项
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/myrag
REDIS_URL=redis://redis:6379/0
MILVUS_HOST=milvus
MILVUS_PORT=19530

# LLM 配置
LLM_PROVIDER=openai        # openai / ollama / vllm
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1

# Embedding 配置
EMBEDDING_PROVIDER=local    # local / openai
EMBEDDING_MODEL=BAAI/bge-m3

# 分块配置
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNK_STRATEGY=recursive    # fixed / recursive / semantic
```

### 10.3 健康检查

```python
@router.get("/health")
async def health_check():
    checks = {
        "database": await check_db(),
        "redis": await check_redis(),
        "milvus": await check_milvus(),
        "llm": await check_llm(),
    }
    status = "healthy" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

---

## 十一、开发路线图

### Phase 1: 基础骨架（Week 1）
- [x] 项目结构搭建、依赖管理
- [ ] 配置管理系统（Pydantic Settings）
- [ ] FastAPI 应用骨架 + 健康检查
- [ ] 数据库模型 + Repository
- [ ] Docker Compose 基础服务

### Phase 2: 文档处理（Week 2）
- [ ] PDF / DOCX / Markdown 解析器
- [ ] 三种分块策略实现
- [ ] Embedding 服务（本地 + API）
- [ ] 向量存储（Milvus + FAISS）
- [ ] 文档上传 API + 异步处理

### Phase 3: 检索与生成（Week 3）
- [ ] Dense / Sparse / Hybrid 检索
- [ ] RRF 融合算法
- [ ] Reranker 集成
- [ ] LLM 服务 + 流式输出
- [ ] Prompt 模板管理
- [ ] 基础问答 API

### Phase 4: 高级特性（Week 4）
- [ ] 多轮对话 + 历史管理
- [ ] 查询改写（HyDE / Multi-Query）
- [ ] 语义缓存
- [ ] Parent-Child 分块策略
- [ ] WebSocket 支持

### Phase 5: 评估与可观测性（Week 5）
- [ ] RAGAS 风格评估框架
- [ ] 结构化日志
- [ ] Prometheus 指标
- [ ] 链路追踪
- [ ] Grafana Dashboard

### Phase 6: 优化与面试准备（Week 6）
- [ ] 性能优化（批处理、并发）
- [ ] Self-RAG / Corrective RAG
- [ ] 完善文档和 README
- [ ] 准备面试演示 Demo
- [ ] 编写技术博客总结

---

## 十二、面试应答准备要点

### 12.1 "为什么选择这个技术栈？" 的回答框架

| 选型 | 理由 | 对比方案 |
|------|------|---------|
| FastAPI vs Flask | 原生 async、自动文档、类型安全 | Flask 生态更大但缺少 async 原生支持 |
| Milvus vs Pinecone | 开源自部署、支持混合检索 | Pinecone 全托管但不开源 |
| BGE-M3 vs OpenAI | 支持多向量、可私有化部署 | OpenAI 质量高但依赖网络 |
| PostgreSQL vs MongoDB | 关系数据 + JSON 灵活性 | MongoDB 更灵活但事务支持弱 |

### 12.2 核心技术问题清单

1. **分块大小对 RAG 效果的影响？** → 小块精准但缺上下文，大块完整但噪声多，Parent-Child 是最佳实践
2. **为什么需要混合检索？** → 语义检索擅长理解意图，关键词检索擅长精确匹配，互补提升效果
3. **Reranker 为什么有效？** → Cross-Encoder 能看到 Query-Doc 交互，比 Bi-Encoder 的独立编码更精确
4. **如何评估 RAG 系统？** → RAGAS 框架：Faithfulness + Relevancy + Precision + Recall
5. **如何处理大文档？** → 流式处理 + 异步任务队列 + 增量索引
6. **语义缓存如何实现？** → 查询 Embedding → 向量相似度检索 → 阈值判断
7. **多轮对话中如何处理指代消解？** → 查询改写，将对话历史融入当前查询

---

## 十三、附录

### A. 参考资料

- [RAG Survey Paper](https://arxiv.org/abs/2312.10997) - Retrieval-Augmented Generation for Large Language Models: A Survey
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG 评估框架
- [LangChain](https://github.com/langchain-ai/langchain) - 参考其 RAG 设计模式（但本项目从零实现）
- [Milvus Documentation](https://milvus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

### B. 本文档版本

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0 | 2026-02-27 | 初始版本，完成整体设计 |



整体设计思路：以 AI 系统工程师面试考点为导向，将所有高频知识点融入到一个完整的 RAG 项目中。
覆盖的核心面试知识域（13大方向）：
RAG 架构演进 — Naive / Advanced / Modular RAG，Self-RAG，Corrective RAG
文档处理 — 多格式解析（PDF/DOCX/MD）、OCR、表格提取
分块策略 — Fixed / Recursive / Semantic 三种策略 + Parent-Child 高级策略
Embedding — Dense / Sparse / Multi-Vector，本地模型与 API 双支持
向量检索 — HNSW/IVF 索引算法、ANN 原理、元数据过滤
混合检索 — Dense + BM25 + RRF 融合，查询改写（HyDE/Multi-Query）
重排序 — Bi-Encoder vs Cross-Encoder，两阶段检索精排
LLM 集成 — 流式输出（SSE/WebSocket）、Token 管理、多模型切换
Prompt 工程 — 模板管理、Few-Shot、CoT
评估体系 — RAGAS 风格五大指标（Faithfulness/Relevancy/Precision/Recall/Correctness）
Python 工程化 — 设计模式（7种）、类型注解、异步编程、依赖注入、错误处理
可观测性 — 结构化日志、Prometheus 指标、链路追踪
DevOps — Docker Compose 一键部署、健康检查、多环境配置
项目分 6 个阶段（约 6 周），从基础骨架到完整可演示的系统，并包含面试应答准备要点。