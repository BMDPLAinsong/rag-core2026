# main.py（多知识库 RAG 版）
import os
import time
import uuid
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ─── 配置 ───
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["rag"]

EMBEDDING_MODEL = cfg["embedding_model"]
LLM_MODEL = cfg["llm_model"]
DB_PATH = cfg["vector_store_path"]
CHUNK_SIZE = cfg["chunk_size"]
CHUNK_OVERLAP = cfg["chunk_overlap"]
TOP_K = cfg["top_k"]
DATA_DIR = cfg["data_dir"]
REGISTRY_PATH = Path(DATA_DIR) / "kb_registry.json"  # 知识库与文档注册表
MAX_HISTORY = 10

os.makedirs(DATA_DIR, exist_ok=True)

# ─── 组件 ───
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL)
chroma_client = chromadb.PersistentClient(path=DB_PATH)


def get_db(kb_id: str) -> Chroma:
    """每个知识库一个独立 collection，实现数据隔离"""
    return Chroma(
        persist_directory=DB_PATH,
        collection_name=f"kb_{kb_id}",
        embedding_function=embeddings,
    )

# ─── 知识库注册表（JSON 持久化） ───
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ─── 会话记忆（按 知识库:会话 隔离） ───
session_stores: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    return session_stores[session_id]

# ─── 数据模型 ───
class KBCreateRequest(BaseModel):
    name: str

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = TOP_K

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[dict]

# ─── 带记忆的 RAG 链（多知识库） ───
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的知识库问答助手。请仅根据下方编号的上下文片段回答问题。
要求：
1. 引用了某个片段的关键信息后，必须紧跟标注该片段的编号，如 [1] 或 [2][3]。
2. 如果上下文不足以回答问题，请明确说明"根据现有资料无法回答"。
3. 结合对话历史理解当前问题中的指代（如"它"、"上面提到的"）。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "上下文片段：\n{context}\n\n问题：{question}"),
])


def rag_func(state: dict) -> dict:
    db = get_db(state["kb_id"])
    docs = db.similarity_search(state["question"], k=state.get("top_k") or TOP_K)
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        parts.append(
            f"[{i}] 来源：{meta.get('source_file', '?')} 第{meta.get('page', '?')}页\n{d.page_content}"
        )
    state["context"] = "\n\n".join(parts)
    answer = (prompt | llm).invoke(state).content
    sources = [
        {
            "index": i + 1,
            "doc_id": d.metadata.get("doc_id", ""),
            "source_file": d.metadata.get("source_file", ""),
            "page": d.metadata.get("page", ""),
            "content": d.page_content,
        }
        for i, d in enumerate(docs)
    ]
    return {"answer": answer, "sources": sources}


rag_chain = RunnableWithMessageHistory(
    RunnableLambda(rag_func),
    get_session_history,
    input_messages_key="question",
    output_messages_key="answer",
    history_messages_key="history",
)

# ─── FastAPI ───
app = FastAPI(title="RAG Knowledge Base")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(session_stores)}


# ─── 知识库管理 ───
@app.get("/kbs")
def list_kbs():
    reg = load_registry()
    return [
        {
            "kb_id": kb_id,
            "name": info["name"],
            "doc_count": len(info.get("documents", {})),
            "created_at": info.get("created_at", ""),
        }
        for kb_id, info in sorted(reg.items(), key=lambda kv: kv[1].get("created_at", ""))
    ]


@app.post("/kbs")
def create_kb(req: KBCreateRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    reg = load_registry()
    for info in reg.values():
        if info["name"] == name:
            raise HTTPException(status_code=400, detail=f"知识库「{name}」已存在")
    kb_id = uuid.uuid4().hex[:12]
    reg[kb_id] = {"name": name, "created_at": now(), "documents": {}}
    save_registry(reg)
    return {"kb_id": kb_id, "name": name}


@app.delete("/kbs/{kb_id}")
def delete_kb(kb_id: str):
    reg = load_registry()
    if kb_id not in reg:
        raise HTTPException(status_code=404, detail="知识库不存在")
    try:
        chroma_client.delete_collection(f"kb_{kb_id}")
    except Exception:
        pass  # collection 尚未创建时忽略
    del reg[kb_id]
    save_registry(reg)
    return {"ok": True}


@app.get("/kbs/{kb_id}/documents")
def list_documents(kb_id: str):
    reg = load_registry()
    if kb_id not in reg:
        raise HTTPException(status_code=404, detail="知识库不存在")
    docs = [
        {"doc_id": doc_id, **info}
        for doc_id, info in reg[kb_id].get("documents", {}).items()
    ]
    docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return docs


@app.delete("/kbs/{kb_id}/documents/{doc_id}")
def delete_document(kb_id: str, doc_id: str):
    reg = load_registry()
    if kb_id not in reg:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if doc_id not in reg[kb_id].get("documents", {}):
        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        get_db(kb_id)._collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    del reg[kb_id]["documents"][doc_id]
    save_registry(reg)
    return {"ok": True}


@app.post("/kbs/{kb_id}/ingest")
async def ingest(kb_id: str, file: UploadFile = File(...)):
    reg = load_registry()
    if kb_id not in reg:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="目前仅支持 PDF 文件")

    doc_id = uuid.uuid4().hex[:12]
    filename = file.filename
    filepath = os.path.join(DATA_DIR, f"{doc_id}_{filename}")

    # 先登记为"解析中"，解析失败状态可被前端看到
    reg[kb_id]["documents"][doc_id] = {
        "filename": filename,
        "chunk_count": 0,
        "pages": 0,
        "status": "parsing",
        "error": None,
        "summary": "",
        "created_at": now(),
    }
    save_registry(reg)

    try:
        with open(filepath, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(filepath)
        documents = loader.load()
        if not documents:
            raise ValueError("未从 PDF 中解析到任何文本内容")
        for doc in documents:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["source_file"] = filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", ""],
        )
        chunks = splitter.split_documents(documents)
        get_db(kb_id).add_documents(chunks)

        full_text = "\n".join(d.page_content for d in documents)
        reg[kb_id]["documents"][doc_id].update({
            "chunk_count": len(chunks),
            "pages": len(documents),
            "status": "ok",
            "summary": full_text[:300],
        })
        save_registry(reg)
        return {"doc_id": doc_id, "filename": filename,
                "chunk_count": len(chunks), "pages": len(documents), "status": "ok"}
    except Exception as e:
        reg[kb_id]["documents"][doc_id].update({
            "status": "failed", "error": str(e)[:200],
        })
        save_registry(reg)
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")

# ─── 问答 ───
@app.post("/kbs/{kb_id}/query", response_model=QueryResponse)
def query(kb_id: str, req: QueryRequest):
    reg = load_registry()
    if kb_id not in reg:
        raise HTTPException(status_code=404, detail="知识库不存在")
    session_id = req.session_id or str(uuid.uuid4())

    history = get_session_history(f"{kb_id}:{session_id}")
    messages = history.messages
    if len(messages) > MAX_HISTORY * 2:
        history.clear()
        for msg in messages[-MAX_HISTORY * 2:]:
            history.add_message(msg)

    result = rag_chain.invoke(
        {"kb_id": kb_id, "question": req.question, "top_k": req.top_k},
        config={"configurable": {"session_id": f"{kb_id}:{session_id}"}},
    )
    return QueryResponse(
        answer=result["answer"],
        session_id=session_id,
        sources=result["sources"],
    )


@app.post("/kbs/{kb_id}/session/{session_id}/clear")
def clear_session(kb_id: str, session_id: str):
    key = f"{kb_id}:{session_id}"
    if key in session_stores:
        session_stores[key].clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
