"""Index vectoriel Qdrant + recherche hybride légère (filtre lexical sur les noms de médicaments)."""
from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .config import settings
from .documents import Chunk
from .llm import Embedder


@dataclass
class Hit:
    score: float
    text: str
    payload: dict


def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


class VectorStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = settings.collection):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def index(self, chunks: list[Chunk], batch_size: int = 64, recreate: bool = True, progress=None) -> None:
        dim = len(self.embedder.embed(["dimension"])[0])
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            with warnings.catch_warnings():  # sans effet (mais inoffensif) en mode embarqué
                warnings.simplefilter("ignore")
                self.client.create_payload_index(self.collection, "keywords", models.PayloadSchemaType.KEYWORD)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.client.upsert(self.collection, points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                    vector=v,
                    payload={**c.payload, "chunk_id": c.id, "text": c.text},
                )
                for c, v in zip(batch, vectors)
            ])
            if progress:
                progress(min(start + batch_size, len(chunks)), len(chunks))

    def _search(self, vector, limit: int, flt=None) -> list[Hit]:
        res = self.client.query_points(self.collection, query=vector, limit=limit,
                                       query_filter=flt, with_payload=True)
        return [Hit(p.score, p.payload["text"], p.payload) for p in res.points]

    def search(self, query: str, k: int = settings.top_k, keywords: list[str] | None = None) -> list[Hit]:
        """Recherche dense ; si la question cite un médicament/une substance connue, on
        récupère d'abord les chunks de ces médicaments (filtre sur les mots-clés), puis on
        complète par la recherche sémantique pure. Fusion par Reciprocal Rank Fusion."""
        vector = self.embedder.embed([query])[0]
        dense = self._search(vector, k * 2)
        if not keywords:
            return dense[:k]
        flt = models.Filter(must=[models.FieldCondition(key="keywords", match=models.MatchAny(any=keywords))])
        filtered = self._search(vector, k * 2, flt)
        return rrf([filtered, dense], k, weights=[2.0, 1.0])


def rrf(rankings: list[list[Hit]], k: int, weights: list[float], c: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking, w in zip(rankings, weights):
        for rank, h in enumerate(ranking):
            key = h.payload["chunk_id"]
            scores[key] = scores.get(key, 0.0) + w / (c + rank + 1)
            hits.setdefault(key, h)
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(scores[i], hits[i].text, hits[i].payload) for i in order]
