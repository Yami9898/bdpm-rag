"""Évaluation de la recherche : Hit@k et MRR, recherche dense seule vs hybride.

    python -m scripts.evaluate                    # eval/questions.jsonl
    python -m scripts.evaluate --k 3

Une question est réussie si un des k chunks retrouvés contient le terme attendu
(nom de médicament ou substance, comparaison sans accents ni casse).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdpm_rag.config import ROOT
from bdpm_rag.documents import normalize, tokens
from bdpm_rag.graph import load_vocabulary
from bdpm_rag.store import VectorStore


def evaluate(store: VectorStore, questions: list[dict], vocabulary: set[str], k: int) -> dict:
    results = {}
    for mode in ("dense", "hybride"):
        hits_at_k, rr = 0, 0.0
        for q in questions:
            kw = sorted(set(tokens(q["question"])) & vocabulary) if mode == "hybride" else None
            found = store.search(q["question"], k=k, keywords=kw)
            target = normalize(q["attendu"])
            rank = next((i for i, h in enumerate(found, 1) if target in normalize(h.text)), None)
            if rank:
                hits_at_k += 1
                rr += 1 / rank
        results[mode] = {"hit@k": hits_at_k / len(questions), "mrr": rr / len(questions)}
    return results


def main() -> None:
    from bdpm_rag.llm import OllamaEmbedder
    from bdpm_rag.store import get_client

    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = VectorStore(get_client(), OllamaEmbedder())
    res = evaluate(store, questions, load_vocabulary(), args.k)
    print(f"{len(questions)} questions, k={args.k}\n")
    print(f"{'mode':<10}{'Hit@k':>8}{'MRR':>8}")
    for mode, m in res.items():
        print(f"{mode:<10}{m['hit@k']:>8.2f}{m['mrr']:>8.2f}")


if __name__ == "__main__":
    main()
