"""文档加载器抽象基类"""
from abc import ABC, abstractmethod
from typing import List

from core.models import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """读取文件并返回 Document 列表"""
        raise NotImplementedError
