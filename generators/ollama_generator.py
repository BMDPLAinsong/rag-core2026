"""Ollama 生成器"""
import ollama

from core.config import settings


SYSTEM_PROMPT = """你是一个基于检索增强生成（RAG）的问答助手。
请严格基于下面提供的上下文回答用户问题。
如果上下文中没有相关信息，请如实回答"根据现有资料无法回答"，不要编造内容。
"""


class OllamaGenerator:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama.chat_model
        self.base_url = base_url or settings.ollama.base_url
        self.client = ollama.Client(host=self.base_url)

    def generate(self, question: str, context: str) -> str:
        prompt = f"上下文：\n{context}\n\n问题：{question}\n\n回答："
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": settings.generation.temperature,
                "num_predict": settings.generation.max_tokens,
            },
        )
        return resp["message"]["content"]
