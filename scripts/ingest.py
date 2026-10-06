"""Ingestion : téléchargement BDPM -> chunks -> embeddings Ollama -> Qdrant.

Exemples :
    python -m scripts.ingest --download            # tout le corpus commercialisé
    python -m scripts.ingest --download --limit 500  # démo rapide
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from bdpm_rag import bdpm
from bdpm_rag.config import settings
from bdpm_rag.documents import build_chunks, build_vocabulary
from bdpm_rag.llm import OllamaEmbedder
from bdpm_rag.store import VectorStore, get_client


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="télécharger les fichiers BDPM")
    p.add_argument("--force", action="store_true", help="re-télécharger même si présents")
    p.add_argument("--limit", type=int, default=None, help="nombre max de spécialités (démo)")
    p.add_argument("--all-status", action="store_true", help="inclure les médicaments non commercialisés")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.download:
        bdpm.download(settings.raw_dir, settings.bdpm_base_url, force=args.force)
    tables = bdpm.load_all(settings.raw_dir)

    chunks = build_chunks(tables, settings.bdpm_fiche_url,
                          only_commercialised=not args.all_status, limit=args.limit)
    vocab = build_vocabulary(chunks)
    settings.vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks, vocabulaire de {len(vocab)} termes")

    t0 = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.time() - t0, 1e-6)
        sys.stdout.write(f"\rEmbeddings : {done}/{total} ({rate:.0f} chunks/s)")
        sys.stdout.flush()

    store = VectorStore(get_client(), OllamaEmbedder())
    store.index(chunks, progress=progress)
    print(f"\nIndex « {settings.collection} » prêt en {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
