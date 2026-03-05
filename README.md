# MyRAG - 智能文档问答系统

> 从零实现的 RAG（Retrieval-Augmented Generation）系统，覆盖 AI 系统工程师面试核心考点。

## 项目亮点

- **全栈实现**：从文档解析到前端对话，全链路手写，无 LangChain 依赖
- **面试导向**：每个模块标注面试考点，覆盖 13 大知识领域
- **7 种设计模式**：工厂、策略、观察者、责任链、单例、Repository、Pipeline
- **三种 RAG 架构**：Naive RAG + Self-RAG + Corrective RAG

## 技术栈

| 层级 | 技术 | 面试考点 |
|------|------|---------|
| Web 框架 | FastAPI | async/await、中间件、依赖注入、SSE/WebSocket |
| 数据库 | SQLAlchemy 2.0 + SQLite | 异步 ORM、关系映射、事务管理 |
| Embedding | sentence-transformers / OpenAI API | 本地模型 vs API、批处理、归一化 |
| 向量存储 | FAISS | IndexFlatIP、IndexIDMap、持久化 |
| 稀疏检索 | BM25 (rank-bm25 + jieba) | TF-IDF、中文分词 |
| 混合检索 | Dense + Sparse + RRF | Reciprocal Rank Fusion 融合算法 |
| LLM | OpenAI 兼容 API | 流式输出、重试策略、多模型切换 |
| 评估 | RAGAS 风格 | Faithfulness / Relevancy / Precision / Recall |
| 监控 | Prometheus + Grafana | Counter / Gauge / Histogram、PromQL |
| 部署 | Docker Compose | 多阶段构建、健康检查、编排 |

## 项目结构

```
my_rag/
├── api/                    # API 层
│   ├── routes/             # 路由（health / knowledge / document / chat / evaluation / metrics / websocket）
│   ├── schemas/            # Pydantic 请求/响应模型
│   └── middleware/         # 中间件（请求追踪 + Prometheus 指标）
├── config/                 # 配置管理（Pydantic Settings）
├── core/                   # 核心业务逻辑
│   ├── rag_pipeline.py     # RAG Pipeline 编排器（Cache → Rewrite → Retrieve → Generate）
│   ├── self_rag.py         # Self-RAG（自反思）
│   ├── corrective_rag.py   # Corrective RAG（纠正式）
│   ├── semantic_cache.py   # 语义缓存
│   ├── document_pipeline.py# 文档处理流水线（Parse → Chunk → Embed → Index）
│   └── dependencies.py     # 依赖注入 / 组件单例管理
├── domain/                 # 领域层
│   ├── embedding/          # Embedding（本地 + OpenAI）
│   ├── retrieval/          # 检索器（Dense + Sparse + Hybrid + QueryRewriter）
│   ├── llm/                # LLM（OpenAI 兼容）
│   ├── chunking/           # 分块策略（Fixed / Recursive / Parent-Child）
│   ├── parser/             # 文档解析器（PDF / DOCX / MD / HTML / TXT）
│   └── prompt/             # Prompt 模板
├── evaluation/             # RAGAS 风格评估框架
├── infrastructure/         # 基础设施层
│   ├── database/           # SQLAlchemy ORM + 会话管理
│   └── vector_store/       # FAISS 向量存储
├── utils/                  # 工具模块
│   ├── logger.py           # 结构化日志（structlog）
│   ├── metrics.py          # Prometheus 指标（11 个指标）
│   ├── tracing.py          # Span 级链路追踪
│   ├── concurrency.py      # 并发控制（Semaphore + 批处理）
│   └── token_counter.py    # Token 计数（tiktoken）
├── templates/              # Jinja2 前端模板
└── static/                 # 前端静态资源（CSS + JS）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key
```

支持的 LLM 服务（任何 OpenAI 兼容接口）：

| 服务 | LLM_BASE_URL | LLM_MODEL |
|------|-------------|-----------|
| 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini |

### 3. 启动服务

```bash
uvicorn my_rag.main:app --reload --host 0.0.0.0 --port 8000
```

首次启动会下载 Embedding 模型（约 100MB）。

### 4. 访问

- 前端：http://localhost:8000
- API 文档：http://localhost:8000/api/docs
- Prometheus 指标：http://localhost:8000/api/v1/metrics

### Docker 部署

```bash
docker-compose up -d
```

启动后访问：
- 应用：http://localhost:8000
- Prometheus：http://localhost:9090
- Grafana：http://localhost:3000（admin/admin）

## 核心面试知识点

### 1. RAG 架构演进

```
Naive RAG:     Query → Retrieve → Generate
Advanced RAG:  Query → [Rewrite] → Retrieve → [Rerank] → Generate
Self-RAG:      Query → [需要检索?] → Retrieve → Generate → [自我评估] → [重试?]
Corrective RAG: Query → Retrieve → [文档打分] → [过滤/补充] → Generate
```

### 2. 混合检索 + RRF 融合

```
Dense (FAISS)  ───→ 排名列表1 ──┐
                                 ├─→ RRF: score(d) = Σ 1/(k + rank_i(d)) → 最终排序
Sparse (BM25)  ───→ 排名列表2 ──┘
```

### 3. 查询改写

- **HyDE**：LLM 生成假设性文档 → 用假设文档做检索（语义更丰富）
- **Multi-Query**：一个问题改写为多个子问题 → 分别检索 → 合并去重

### 4. 分块策略

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| Fixed | 固定窗口 + 重叠 | 简单场景 |
| Recursive | 按自然分隔符递归分割 | 通用场景（默认） |
| Parent-Child | 小块检索 + 大块返回 | 需要上下文的场景 |

### 5. 评估指标（RAGAS 风格）

| 指标 | 评估什么 |
|------|---------|
| Faithfulness | 回答是否忠实于文档（无幻觉） |
| Answer Relevancy | 回答是否和问题相关 |
| Context Precision | 检索排序质量 |
| Context Recall | 标准答案被文档覆盖的程度 |
| Answer Correctness | 回答与标准答案的一致性 |

## API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/health | 健康检查 |
| POST | /api/v1/knowledge-bases | 创建知识库 |
| POST | /api/v1/knowledge-bases/{id}/documents | 上传文档 |
| POST | /api/v1/chat/completions | 对话（SSE 流式） |
| WS | /api/v1/ws/chat | WebSocket 对话 |
| POST | /api/v1/evaluations | 运行评估 |
| GET | /api/v1/metrics | Prometheus 指标 |

## 开发路线

- [x] Phase 1: 基础骨架 + 前端
- [x] Phase 2: 文档解析 + 分块
- [x] Phase 3: Embedding + 向量存储 + 混合检索 + LLM
- [x] Phase 4: 查询改写 + 语义缓存 + Parent-Child + WebSocket
- [x] Phase 5: RAGAS 评估 + Prometheus + 链路追踪
- [x] Phase 6: Self-RAG + Corrective RAG + Docker 部署
