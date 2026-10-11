"""统一数据模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """原始文档"""
    source: str = Field(..., description="文档来源，如文件名")
    text: str = Field(..., description="文档全文")
    metadata: dict = Field(default_factory=dict, description="附加元数据，如页码")


class Chunk(BaseModel):
    """文档切片"""
    id: str = Field(..., description="切片唯一ID")
    text: str = Field(..., description="切片文本")
    metadata: dict = Field(default_factory=dict, description="切片元数据")
    embedding: Optional[List[float]] = None


class QueryResult(BaseModel):
    """检索结果"""
    text: str
    metadata: dict
    distance: float


class RAGResponse(BaseModel):
    """RAG 最终响应"""
    answer: str
    sources: List[dict] = Field(default_factory=list)
