"""Markdown 加载器"""
from pathlib import Path
from typing import List

from core.models import Document
from loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [
            Document(
                source=path.name,
                text=text,
                metadata={"file_type": "markdown"},
            )
        ]
