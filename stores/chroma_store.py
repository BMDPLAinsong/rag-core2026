"""ChromaDB 向量存储"""
from typing import List

import chromadb

from core.config import settings
from core.models import Chunk


class ChromaStore:
    def __init__(self, persist_directory: str | None = None, collection_name: str | None = None):
        persist_directory = persist_directory or settings.chroma.persist_directory
        collection_name = collection_name or settings.chroma.collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], top_k: int) -> List[dict]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        # Chroma 返回结构: ids, distances, documents, metadatas
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            {"id": _id, "text": doc, "metadata": meta, "distance": dist}
            for _id, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]
