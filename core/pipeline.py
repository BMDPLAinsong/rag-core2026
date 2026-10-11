"""核心 Pipeline 编排"""
from typing import List

from core.models import Document, RAGResponse
from chunkers.recursive_chunker import RecursiveChunker
from embedders.ollama_embedder import OllamaEmbedder
from generators.ollama_generator import OllamaGenerator
from loaders.base import BaseLoader
from preprocessors.text_cleaner import TextCleaner
from retrievers.vector_retriever import VectorRetriever
from stores.chroma_store import ChromaStore


class RAGPipeline:
    """RAG 全流程编排：加载 -> 清洗 -> 切片 -> 向量化 -> 存储 -> 检索 -> 生成"""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.chunker = RecursiveChunker()
        self.embedder = OllamaEmbedder()
        self.store = ChromaStore()
        self.retriever = VectorRetriever(self.store)
        self.generator = OllamaGenerator()

    def ingest(self, loader: BaseLoader, file_path: str) -> int:
        """将文档加载、清洗、切片、向量化并存入向量库"""
        documents: List[Document] = loader.load(file_path)
        cleaned = [self.cleaner.clean(doc) for doc in documents]
        chunks = self.chunker.split(cleaned)
        embeddings = self.embedder.embed([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self.store.add(chunks)
        return len(chunks)

    def query(self, question: str) -> RAGResponse:
        """检索 + 生成回答"""
        results = self.retriever.retrieve(question)
        context = "\n\n".join(r.text for r in results)
        answer = self.generator.generate(question, context)
        return RAGResponse(
            answer=answer,
            sources=[{"text": r.text, "metadata": r.metadata} for r in results],
        )
