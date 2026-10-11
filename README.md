# RAG-Core — 基于 Ollama 的本地 RAG 问答服务

一个轻量级的本地检索增强生成（RAG）服务：上传 PDF 文档，自动解析、切片、向量化并存入 Chroma，然后通过 API 进行基于文档内容的智能问答。全部组件（嵌入模型、大语言模型、向量库）均在本地运行，数据不出本机。

## 技术栈

| 组件 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| 编排 | LangChain（community / ollama / chroma / text-splitters） |
| 嵌入模型 | Ollama `nomic-embed-text` |
| 对话模型 | Ollama（默认 `SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU`，可在 config.yaml 中更换） |
| 向量库 | ChromaDB（持久化到 `./data/chroma`） |

## 环境要求

- Python 3.11+
- [Ollama](https://ollama.com/) 已安装并运行（默认 `http://localhost:11434`）
- 已拉取所需模型：

```bash
ollama pull nomic-embed-text
ollama pull <你在 config.yaml 中配置的对话模型>
```

## 安装与启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务（默认 0.0.0.0:8000）
python main.py
```

启动后访问交互式 API 文档：http://localhost:8000/docs

## 配置说明（config.yaml）

| 配置节 | 键 | 说明 | 默认值 |
|---|---|---|---|
| `app` | `host` / `port` | 服务监听地址与端口 | `0.0.0.0:8000` |
| `ollama` | `base_url` | Ollama 服务地址 | `http://localhost:11434` |
| | `embed_model` | 嵌入模型名 | `nomic-embed-text` |
| | `chat_model` | 对话模型名 | — |
| `chroma` | `persist_directory` | 向量库持久化目录 | `./data/chroma` |
| | `collection_name` | 集合名 | `rag_docs` |
| `chunking` | `chunk_size` / `chunk_overlap` | 切片大小与重叠字符数 | `512 / 50` |
| | `separators` | 递归切片分隔符 | 中文标点优先 |
| `retrieval` | `top_k` | 检索返回片段数 | `4` |
| `generation` | `temperature` / `max_tokens` | 生成温度与最大输出 token | `0.7 / 2048` |

配置由 [core/config.py](core/config.py) 中的 pydantic `Settings` 加载，所有键均有默认值，缺省时可安全省略。

## API 接口

### `GET /health`

健康检查。

```powershell
curl.exe http://localhost:8000/health
# {"status":"ok"}
```

### `POST /ingest`

上传 PDF 文档，解析 → 切片 → 向量化 → 入库。

- 参数：`multipart/form-data`，字段名 `file`
- 返回：`doc_id`、`filename`、`chunk_count`

```powershell
curl.exe -X POST http://localhost:8000/ingest -F "file=@你的文档.pdf"
# {"doc_id":"e332c6490fa5","filename":"你的文档.pdf","chunk_count":12}
```

### `POST /query`

检索相关片段并调用本地 LLM 生成回答。

- 请求体：`{"question": "问题", "top_k": 4}`（`top_k` 可省略，默认取配置值）
- 返回：`answer`（回答文本）、`sources`（引用片段：来源文件、页码、内容摘要）

```powershell
$body = @{"question" = "血压平均降低了多少？"} | ConvertTo-JSON
Invoke-RestMethod -Uri http://localhost:8000/query -Method Post `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
  -ContentType "application/json"
```

> 当向量库为空或检索不到相关内容时，服务会明确回答"根据现有资料无法回答"，避免模型编造答案。

## 目录结构

```
rag-core/
├── main.py                 # 独立版入口（LangChain 实现，当前主入口）
├── config.yaml             # 全部配置项
├── requirements.txt        # 依赖清单
├── core/
│   ├── config.py           # pydantic 配置加载（main.py 复用）
│   ├── pipeline.py         # RAGPipeline：加载→清洗→切片→嵌入→存储→检索→生成
│   └── models.py           # Document / Chunk / QueryResult / RAGResponse 数据模型
├── loaders/                # PDF / DOCX / Markdown 加载器（模块化版）
├── preprocessors/          # 文本清洗
├── chunkers/               # 递归切片器
├── embedders/              # Ollama 嵌入封装
├── stores/                 # Chroma 存储封装
├── retrievers/             # 向量检索封装
├── generators/             # Ollama 生成封装
├── api/
│   └── routes.py           # 模块化版路由（支持 pdf/docx/md，挂载后可用）
└── data/
    ├── chroma/             # 向量库持久化数据
    └── uploads/            # 上传文档保存目录（运行时自动创建）
```

## 关于两套实现

项目当前包含两条实现路径：

1. **`main.py`（推荐使用）**：基于 LangChain 生态的独立 FastAPI 应用，开箱即用，已验证可用。
2. **模块化包（`core/` + `loaders/` + `api/` 等）**：自研分层的 RAGPipeline 实现，职责更细、便于扩展（如支持 docx/md），但入口尚未与 `main.py` 合并，需自行将 `api.router` 挂载到 app 后使用。

## 常见问题

- **启动报错 `[Errno 10048]` 端口被占用**：已有实例在运行，先结束旧进程（`Get-NetTCPConnection -LocalPort 8000` 查 PID 后 `Stop-Process`），或修改 config.yaml 的端口。
- **接口报连接错误 / 模型不存在**：确认 Ollama 已启动（`curl.exe http://localhost:11434/api/tags` 可查看已安装模型），且 config.yaml 中的模型名与之一致。
- **中文问题显示乱码**：PowerShell 控制台显示编码问题，服务端返回为标准 UTF-8 JSON，用 `python` 或支持 UTF-8 的客户端解析即可。
