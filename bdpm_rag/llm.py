"""Clients Ollama (embeddings + génération), 100 % locaux."""
from __future__ import annotations

from typing import Protocol

from .config import settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class OllamaEmbedder:
    def __init__(self, model: str = settings.embed_model, host: str = settings.ollama_host):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return list(self.client.embed(model=self.model, input=texts)["embeddings"])


class OllamaChat:
    def __init__(self, model: str = settings.llm_model, host: str = settings.ollama_host):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0},
        )
        return resp["message"]["content"]
