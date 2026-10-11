"""递归字符切片器（不依赖 langchain，纯手写实现）"""
import hashlib
from typing import List

from core.config import settings
from core.models import Chunk, Document


class RecursiveChunker:
    """按分隔符递归切分，保证每个 chunk 不超过 chunk_size，相邻 chunk 重叠 chunk_overlap"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None,
                 separators: List[str] | None = None):
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.separators = separators or settings.chunking.separators

    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            pieces = self._split_text(doc.text, self.separators)
            for i, piece in enumerate(pieces):
                chunk_id = self._make_id(doc.source, doc.metadata, i)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=piece,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        # 找到第一个能切分文本的分隔符
        sep = separators[-1]  # 默认单字符
        for s in separators:
            if s == "":
                continue
            if s in text:
                sep = s
                break

        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        # 合并小段，控制长度
        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # 如果单段本身超过 chunk_size，递归用更细的分隔符切
                if len(part) > self.chunk_size:
                    sub_seps = separators[separators.index(sep) + 1:] if sep in separators else []
                    merged.extend(self._split_text(part, sub_seps or [""]))
                    current = ""
                else:
                    current = part
        if current:
            merged.append(current)

        # 加 overlap：相邻 chunk 共享末尾字符
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped: List[str] = []
            for i, m in enumerate(merged):
                if i == 0:
                    overlapped.append(m)
                else:
                    tail = merged[i - 1][-self.chunk_overlap:]
                    overlapped.append(tail + m)
            return overlapped
        return merged

    @staticmethod
    def _make_id(source: str, metadata: dict, index: int) -> str:
        raw = f"{source}:{metadata.get('page', 0)}:{index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
