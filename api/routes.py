"""API 路由"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from core.pipeline import RAGPipeline
from loaders.docx_loader import DocxLoader
from loaders.markdown_loader import MarkdownLoader
from loaders.pdf_loader import PDFLoader

router = APIRouter()
pipeline = RAGPipeline()

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PDFLoader()
    elif ext in (".docx", ".doc"):
        return DocxLoader()
    elif ext in (".md", ".markdown"):
        return MarkdownLoader()
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """上传文档并写入向量库"""
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    loader = get_loader(str(save_path))
    count = pipeline.ingest(loader, str(save_path))
    return {"status": "ok", "chunks": count, "file": file.filename}


@router.post("/query")
def query(req: QueryRequest):
    """RAG 问答"""
    if req.top_k is not None:
        pipeline.retriever.top_k = req.top_k
    result = pipeline.query(req.question)
    return result.model_dump()
