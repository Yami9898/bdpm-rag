"""Interroger le RAG en ligne de commande.

    python -m scripts.ask "Quelles sont les conditions de prescription d'Ozempic ?"
"""
from __future__ import annotations

import sys

from bdpm_rag.graph import default_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    result = default_pipeline().invoke({"question": " ".join(sys.argv[1:])})
    print(result["answer"])
    for w in result.get("warnings", []):
        print(f"\n⚠ {w}")
    print("\nSources :")
    for i, h in enumerate(result.get("hits", []), start=1):
        print(f"  [{i}] {h.payload['denomination']} ({h.payload['type']}) — {h.payload['url']}")


if __name__ == "__main__":
    main()
