"""Téléchargement et parsing des fichiers de la Base de Données Publique des Médicaments (BDPM).

Format officiel : fichiers texte, une ligne par enregistrement, colonnes séparées par des
tabulations, sans ligne d'en-tête. Voir « Contenu et format des fichiers téléchargeables
dans la BDM » sur https://base-donnees-publique.medicaments.gouv.fr/telechargement
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Nom de fichier -> colonnes, dans l'ordre de la documentation officielle
SCHEMAS: dict[str, list[str]] = {
    "CIS_bdpm.txt": [
        "cis", "denomination", "forme", "voies", "statut_amm", "procedure_amm",
        "etat_commercialisation", "date_amm", "statut_bdm", "num_autorisation_eu",
        "titulaires", "surveillance_renforcee",
    ],
    "CIS_COMPO_bdpm.txt": [
        "cis", "element", "code_substance", "substance", "dosage",
        "reference_dosage", "nature", "num_liaison",
    ],
    "CIS_HAS_SMR_bdpm.txt": [
        "cis", "code_dossier_has", "motif", "date_avis", "valeur_smr", "libelle_smr",
    ],
    "CIS_CPD_bdpm.txt": ["cis", "condition"],
    "CIS_GENER_bdpm.txt": ["id_groupe", "libelle_groupe", "cis", "type_generique", "tri"],
}

TYPE_GENERIQUE = {"0": "princeps", "1": "générique", "2": "générique par complémentarité posologique",
                  "4": "générique substituable"}


def download(raw_dir: Path, base_url: str, force: bool = False) -> None:
    """Télécharge les fichiers BDPM nécessaires dans ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in SCHEMAS:
        dest = raw_dir / name
        if dest.exists() and not force:
            log.info("Déjà présent : %s", dest)
            continue
        url = f"{base_url}/{name}"
        log.info("Téléchargement %s", url)
        resp = requests.get(url, timeout=120, headers={"User-Agent": "bdpm-rag/0.1"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)


def _decode(raw: bytes) -> str:
    # Les fichiers BDPM sont publiés en UTF-8 ou en Windows-1252 (apostrophe typographique 0x92,
    # « œ » 0x9C…) ; latin-1 en dernier recours pour les octets non définis en cp1252.
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Lit un fichier BDPM de manière tolérante (colonnes manquantes ou en trop, espaces)."""
    rows = []
    n = len(columns)
    for line in _decode(path.read_bytes()).splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split("\t")]
        fields = (fields + [""] * n)[:n]
        rows.append(fields)
    return pd.DataFrame(rows, columns=columns, dtype=str)


def load_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, cols in SCHEMAS.items():
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable — lancez d'abord `python -m scripts.ingest --download`")
        tables[name] = read_table(path, cols)
        log.info("%s : %d lignes", name, len(tables[name]))
    return tables
