"""向量检索器"""
from typing import List

from core.config import settings
from core.models import QueryResult
from embedders.ollama_embedder import OllamaEmbedder
from stores.chroma_store import ChromaStore


class VectorRetriever:
    def __init__(self, store: ChromaStore, top_k: int | None = None):
        self.store = store
        self.top_k = top_k or settings.retrieval.top_k
        self.embedder = OllamaEmbedder()

    def retrieve(self, query: str) -> List[QueryResult]:
        embeddings = self.embedder.embed([query])
        if not embeddings:
            return []
        results = self.store.query(embeddings[0], self.top_k)
        return [
            QueryResult(text=r["text"], metadata=r["metadata"], distance=r["distance"])
            for r in results
        ]
