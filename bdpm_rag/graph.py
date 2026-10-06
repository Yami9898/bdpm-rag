"""Orchestration du RAG avec LangGraph.

    question
       │
  [entites] ── repère les médicaments / substances cités (vocabulaire BDPM)
       │
  [recherche] ── Qdrant : dense + filtre lexical, fusion RRF
       │
   ┌───┴─────────── aucun document ──► [abstention]
  [generation] ── LLM local, réponse uniquement à partir du contexte, citations [n]
       │
  [verification] ── contrôle des citations (références existantes, pas de réponse non sourcée)
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .documents import tokens
from .llm import ChatModel
from .store import Hit, VectorStore

ABSTENTION = ("Je ne trouve pas d'information suffisante dans la Base de Données Publique "
              "des Médicaments pour répondre à cette question.")

SYSTEM_PROMPT = f"""Tu es un assistant d'information sur le médicament destiné aux professionnels de santé.
Règles impératives :
- Réponds UNIQUEMENT à partir des extraits numérotés fournis (issus de la BDPM). N'utilise aucune connaissance externe.
- Chaque affirmation factuelle doit être suivie de sa source entre crochets, par ex. [1] ou [2][3].
- Si les extraits ne permettent pas de répondre, réponds exactement : « {ABSTENTION} »
- Ne fabrique jamais de posologie, d'indication ou d'interaction absente des extraits.
- Réponds en français, de façon concise et structurée."""


class RagState(TypedDict, total=False):
    question: str
    keywords: list[str]
    hits: list[Hit]
    answer: str
    warnings: list[str]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))


def build_graph(store: VectorStore, llm: ChatModel, vocabulary: set[str], k: int = settings.top_k):
    def entites(state: RagState) -> RagState:
        return {"keywords": sorted(set(tokens(state["question"])) & vocabulary)}

    def recherche(state: RagState) -> RagState:
        return {"hits": store.search(state["question"], k=k, keywords=state.get("keywords"))}

    def generation(state: RagState) -> RagState:
        user = f"Extraits de la BDPM :\n\n{format_context(state['hits'])}\n\nQuestion : {state['question']}"
        return {"answer": llm.chat(SYSTEM_PROMPT, user).strip()}

    def abstention(state: RagState) -> RagState:
        return {"answer": ABSTENTION, "warnings": []}

    def verification(state: RagState) -> RagState:
        answer, n = state["answer"], len(state["hits"])
        warnings = []
        if ABSTENTION[:40] in answer:
            return {"warnings": warnings}
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
        if not cited:
            warnings.append("La réponse ne cite aucune source : à vérifier sur les fiches ci-dessous.")
        invalid = sorted(c for c in cited if c < 1 or c > n)
        if invalid:
            warnings.append(f"Citations invalides détectées : {invalid}.")
        return {"warnings": warnings}

    g = StateGraph(RagState)
    g.add_node("entites", entites)
    g.add_node("recherche", recherche)
    g.add_node("generation", generation)
    g.add_node("abstention", abstention)
    g.add_node("verification", verification)
    g.set_entry_point("entites")
    g.add_edge("entites", "recherche")
    g.add_conditional_edges("recherche", lambda s: "generation" if s["hits"] else "abstention",
                            {"generation": "generation", "abstention": "abstention"})
    g.add_edge("generation", "verification")
    g.add_edge("verification", END)
    g.add_edge("abstention", END)
    return g.compile()


def load_vocabulary() -> set[str]:
    if settings.vocab_path.exists():
        return set(json.loads(settings.vocab_path.read_text(encoding="utf-8")))
    return set()


def default_pipeline():
    """Construit le pipeline complet à partir de l'index local et d'Ollama."""
    from .llm import OllamaChat, OllamaEmbedder
    from .store import get_client

    store = VectorStore(get_client(), OllamaEmbedder())
    return build_graph(store, OllamaChat(), load_vocabulary())
