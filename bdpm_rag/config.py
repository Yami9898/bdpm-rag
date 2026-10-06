"""Configuration centralisée (surchargée par variables d'environnement / fichier .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv est optionnel
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("BDPM_DATA_DIR", ROOT / "data"))
    bdpm_base_url: str = os.getenv(
        "BDPM_BASE_URL", "https://base-donnees-publique.medicaments.gouv.fr/download/file"
    )
    bdpm_fiche_url: str = os.getenv(
        "BDPM_FICHE_URL",
        "https://base-donnees-publique.medicaments.gouv.fr/extrait.php?specid={cis}",
    )

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.getenv("EMBED_MODEL", "bge-m3")
    llm_model: str = os.getenv("LLM_MODEL", "mistral")

    # Vide => Qdrant en mode embarqué (fichiers locaux, pas de serveur)
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    collection: str = os.getenv("QDRANT_COLLECTION", "bdpm")

    top_k: int = int(os.getenv("TOP_K", "6"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def vocab_path(self) -> Path:
        return self.data_dir / "vocabulaire.json"


settings = Settings()
