"""文本清洗器"""
import re

from core.models import Document


class TextCleaner:
    """去除多余空白、控制字符、统一换行"""

    _control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _multi_space = re.compile(r"[ \t]{2,}")
    _multi_newline = re.compile(r"\n{3,}")

    def clean(self, doc: Document) -> Document:
        text = doc.text
        text = self._control_chars.sub("", text)
        text = self._multi_space.sub(" ", text)
        text = self._multi_newline.sub("\n\n", text)
        text = text.strip()
        return Document(source=doc.source, text=text, metadata=doc.metadata)
