"""Ollama 向量化器"""
from typing import List

import ollama

from core.config import settings


class OllamaEmbedder:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama.embed_model
        self.base_url = base_url or settings.ollama.base_url
        self.client = ollama.Client(host=self.base_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self.client.embed(model=self.model, input=texts)
        embeddings = resp.get("embeddings", [])
        return embeddings
