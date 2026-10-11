"""配置加载"""
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


class AppSettings(BaseModel):
    name: str = "rag-core"
    host: str = "0.0.0.0"
    port: int = 8000


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"


class ChromaSettings(BaseModel):
    persist_directory: str = "./data/chroma"
    collection_name: str = "rag_docs"


class ChunkingSettings(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class RetrievalSettings(BaseModel):
    top_k: int = 4


class GenerationSettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 2048


class Settings(BaseModel):
    app: AppSettings = AppSettings()
    ollama: OllamaSettings = OllamaSettings()
    chroma: ChromaSettings = ChromaSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()


def load_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        return Settings()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)


settings = load_settings()
