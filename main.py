# main.py（多轮对话版）
import os
import uuid
import yaml
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
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
MAX_HISTORY = 10  # 最多保留最近 10 轮

os.makedirs(DATA_DIR, exist_ok=True)

# ─── 组件 ───
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL)

def get_db():
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# ─── 会话记忆存储（内存，按 session_id 隔离）───
session_stores: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    return session_stores[session_id]

# ─── 数据模型 ───
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = TOP_K

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[dict]

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int

# ─── 带记忆的 RAG 链 ───
def build_rag_chain(top_k: int = TOP_K):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的问答助手。请仅根据以下上下文回答问题。
如果上下文不足以回答问题，请明确说明"根据现有资料无法回答"。
注意结合对话历史理解用户当前问题中的指代（如"它"、"上面提到的"）。"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "上下文：\n{context}\n\n问题：{question}"),
    ])

    def rag_func(state: dict) -> dict:
        db = get_db()
        docs = db.similarity_search(state["question"], k=state.get("top_k") or top_k)
        context = "\n\n---\n\n".join([
            f"[{d.metadata.get('source_file', '?')} 第{d.metadata.get('page', '?')}页]\n{d.page_content}"
            for d in docs
        ])
        state["context"] = context
        answer = (prompt | llm).invoke(state).content
        return {"answer": answer, "docs": docs}

    return RunnableLambda(rag_func)

# 用 RunnableWithMessageHistory 包装
rag_chain = RunnableWithMessageHistory(
    build_rag_chain(),
    get_session_history,
    input_messages_key="question",
    output_messages_key="answer",
    history_messages_key="history",
)

# ─── FastAPI ───
app = FastAPI(title="RAG Core")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    filename = file.filename
    doc_id = uuid.uuid4().hex[:12]
    filepath = os.path.join(DATA_DIR, f"{doc_id}_{filename}")
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    loader = PyPDFLoader(filepath)
    documents = loader.load()
    for doc in documents:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["source_file"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", ""]
    )
    chunks = splitter.split_documents(documents)
    db = get_db()
    db.add_documents(chunks)

    return IngestResponse(doc_id=doc_id, filename=filename, chunk_count=len(chunks))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    # 裁剪历史，避免上下文过长
    history = get_session_history(session_id)
    messages = history.messages
    if len(messages) > MAX_HISTORY * 2:
        history.clear()
        for msg in messages[-MAX_HISTORY * 2:]:
            history.add_message(msg)

    result = rag_chain.invoke(
        {"question": req.question, "top_k": req.top_k},
        config={"configurable": {"session_id": session_id}},
    )
    answer, docs = result["answer"], result["docs"]

    sources = [
        {
            "source_file": d.metadata.get("source_file", ""),
            "page": d.metadata.get("page", ""),
            "content": d.page_content[:200],
        }
        for d in docs
    ]

    return QueryResponse(answer=answer, session_id=session_id, sources=sources)

@app.post("/session/{session_id}/clear")
def clear_session(session_id: str):
    if session_id in session_stores:
        session_stores[session_id].clear()
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(session_stores)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)