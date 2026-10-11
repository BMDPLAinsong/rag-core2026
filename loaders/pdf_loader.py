"""PDF 加载器（基于 PyMuPDF / fitz）"""
from pathlib import Path
from typing import List

import fitz

from core.models import Document
from loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        doc = fitz.open(path)
        documents: List[Document] = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if not text:
                continue
            documents.append(
                Document(
                    source=path.name,
                    text=text,
                    metadata={"page": page_num + 1, "file_type": "pdf"},
                )
            )
        doc.close()
        return documents
