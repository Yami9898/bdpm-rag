"""Tests hors-ligne : parsing BDPM, chunking, recherche hybride et graphe LangGraph,
avec un embedder et un LLM factices (pas besoin d'Ollama)."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from bdpm_rag import bdpm
from bdpm_rag.documents import brand_words, build_chunks, build_vocabulary, normalize, tokens
from bdpm_rag.graph import ABSTENTION, build_graph
from bdpm_rag.store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.org/fiche/{cis}"


class HashEmbedder:
    """Sac de mots haché : déterministe, suffisant pour tester la mécanique."""

    dim = 256

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[A-Z0-9]+", normalize(t)):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM:
    def __init__(self, reply="Réponse fondée [1]."):
        self.reply, self.last_user = reply, None

    def chat(self, system, user):
        self.last_user = user
        return self.reply


@pytest.fixture(scope="module")
def tables():
    return bdpm.load_all(FIXTURES)


@pytest.fixture(scope="module")
def chunks(tables):
    return build_chunks(tables, URL)


@pytest.fixture()
def store(chunks):
    s = VectorStore(QdrantClient(":memory:"), HashEmbedder(), collection="test")
    s.index(chunks)
    return s


def test_parsing_latin1_et_colonnes(tables):
    cis = tables["CIS_bdpm.txt"]
    assert len(cis) == 5
    assert cis.iloc[0]["denomination"] == "DOLIPRANE 1000 mg, comprimé"  # accents décodés
    assert list(cis.columns) == bdpm.SCHEMAS["CIS_bdpm.txt"]


def test_chunks_fiche_et_smr(chunks):
    ids = {c.id for c in chunks}
    assert "60001005:fiche" not in ids  # non commercialisé exclu
    assert {"60001001:fiche", "60001002:smr:0", "60001002:smr:1", "60001003:smr:0"} <= ids
    ozempic = next(c for c in chunks if c.id == "60001003:fiche")
    assert "SÉMAGLUTIDE 1 mg" in ozempic.text
    assert "prescription initiale hospitalière" in ozempic.text
    assert "surveillance renforcée" in ozempic.text
    assert ozempic.payload["url"] == URL.format(cis="60001003")
    generique = next(c for c in chunks if c.id == "60001004:fiche")
    assert "EXCIPIENT" not in generique.text  # seules les substances actives
    assert "le générique" in generique.text
    # l'avis SMR le plus récent vient en premier
    assert "15/06/2019" in next(c for c in chunks if c.id == "60001002:smr:0").text


def test_mots_cles(chunks):
    assert brand_words("XARELTO 20 mg, comprimé pelliculé") == ["XARELTO"]
    assert tokens("Qu'est-ce que l'Ozempic ?") == ["OZEMPIC"]
    vocab = build_vocabulary(chunks)
    assert {"OZEMPIC", "SEMAGLUTIDE", "RIVAROXABAN", "PARACETAMOL"} <= set(vocab)


def test_recherche_hybride_priorise_le_medicament_cite(store):
    hits = store.search("conditions de prescription OZEMPIC", k=3, keywords=["OZEMPIC"])
    assert hits[0].payload["cis"] == "60001003"


def test_graphe_bout_en_bout(store, chunks):
    llm = FakeLLM()
    graph = build_graph(store, llm, set(build_vocabulary(chunks)), k=3)
    out = graph.invoke({"question": "Quel est le SMR de Xarelto ?"})
    assert out["keywords"] == ["XARELTO"]
    assert out["hits"][0].payload["cis"] == "60001002"
    assert "[1]" in llm.last_user and "Question : Quel est le SMR de Xarelto ?" in llm.last_user
    assert out["warnings"] == []


def test_verification_des_citations(store, chunks):
    vocab = set(build_vocabulary(chunks))
    out = build_graph(store, FakeLLM("Sans source."), vocab, k=2).invoke({"question": "Xarelto ?"})
    assert any("aucune source" in w for w in out["warnings"])
    out = build_graph(store, FakeLLM("Faux [9]."), vocab, k=2).invoke({"question": "Xarelto ?"})
    assert any("invalides" in w for w in out["warnings"])


def test_abstention_si_aucun_document(chunks):
    empty = VectorStore(QdrantClient(":memory:"), HashEmbedder(), collection="vide")
    empty.index([])
    out = build_graph(empty, FakeLLM(), set()).invoke({"question": "Ozempic ?"})
    assert out["answer"] == ABSTENTION
