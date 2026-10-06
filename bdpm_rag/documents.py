"""Transformation des tables BDPM en chunks textuels prêts à indexer.

Stratégie de chunking (une spécialité = une unité de sens) :
  * 1 chunk « fiche » par spécialité : identité, composition, prescription, générique ;
  * 1 chunk par avis SMR de la HAS (les 3 plus récents), préfixé par le nom du médicament
    (« contextual chunk header ») pour qu'il reste interprétable isolément.
Chaque chunk porte des métadonnées (CIS, mots-clés normalisés, URL source) pour le filtrage
et la citation.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from .bdpm import TYPE_GENERIQUE

MAX_SMR_CHARS = 1500
SMR_PER_DRUG = 3

# Mots trop génériques pour servir d'ancre lexicale
STOPWORDS = {
    "ACIDE", "CHLORHYDRATE", "SODIQUE", "SODIUM", "POTASSIUM", "CALCIUM", "MONOHYDRATE",
    "DIHYDRATE", "ANHYDRE", "BASE", "SULFATE", "MALEATE", "TARTRATE", "ACETATE",
    "PHOSPHATE", "CITRATE", "HEMIHYDRATE", "TRIHYDRATE", "BROMHYDRATE", "MESILATE",
    "SOLUTION", "COMPRIME", "GELULE", "SIROP", "POUDRE", "INJECTABLE", "BUVABLE",
    "PELLICULE", "SUSPENSION", "CREME", "POMMADE", "COLLYRE", "ENFANT", "ADULTE",
    "NOURRISSON", "LIBERATION", "PROLONGEE", "MODIFIEE", "ORALE", "VOIE", "POUR",
    "AVEC", "SANS", "SUCRE", "ARROW", "BIOGARAN", "SANDOZ", "TEVA", "MYLAN", "VIATRIS",
    "ZENTIVA", "CRISTERS", "ZYDUS", "ACCORD", "EG", "ALMUS", "QUELLE", "QUELS", "QUELLES",
    "MEDICAMENT", "MEDICAMENTS", "CONTIENT", "CONTIENNENT", "EST-CE", "PEUT", "DOSAGE",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.upper()


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", normalize(text)) if len(t) >= 4 and t not in STOPWORDS
            and not t.isdigit()]


def brand_words(denomination: str) -> list[str]:
    """Mots du nom commercial : ce qui précède le premier dosage ou la virgule."""
    head = re.split(r",|\s\d", denomination, maxsplit=1)[0]
    return tokens(head)[:3]


@dataclass
class Chunk:
    id: str
    text: str
    payload: dict = field(default_factory=dict)


def _fmt_date_has(d: str) -> str:
    return f"{d[6:8]}/{d[4:6]}/{d[0:4]}" if len(d) == 8 and d.isdigit() else d


def build_chunks(tables: dict[str, pd.DataFrame], fiche_url: str,
                 only_commercialised: bool = True, limit: int | None = None) -> list[Chunk]:
    cis = tables["CIS_bdpm.txt"]
    if only_commercialised:
        cis = cis[cis["etat_commercialisation"].str.startswith("Commercialis")]
    if limit:
        cis = cis.head(limit)

    compo = tables["CIS_COMPO_bdpm.txt"]
    compo = compo[compo["nature"] == "SA"].groupby("cis")
    cpd = tables["CIS_CPD_bdpm.txt"].groupby("cis")["condition"].apply(list)
    gener = tables["CIS_GENER_bdpm.txt"].set_index("cis")
    gener = gener[~gener.index.duplicated()]
    smr = tables["CIS_HAS_SMR_bdpm.txt"].sort_values("date_avis", ascending=False).groupby("cis")

    compo_groups = set(compo.groups)
    smr_groups = set(smr.groups)
    chunks: list[Chunk] = []

    for row in cis.itertuples(index=False):
        code = row.cis
        substances = []
        if code in compo_groups:
            for s in compo.get_group(code).itertuples(index=False):
                dose = f" {s.dosage}" if s.dosage else ""
                ref = f" (pour {s.reference_dosage})" if s.reference_dosage else ""
                substances.append(f"{s.substance}{dose}{ref}")
        substances = list(dict.fromkeys(substances))

        keywords = set(brand_words(row.denomination))
        for s in substances:
            keywords.update(tokens(s.split(" (")[0]))

        url = fiche_url.format(cis=code)
        base_payload = {
            "cis": code,
            "denomination": row.denomination,
            "url": url,
            "keywords": sorted(keywords),
            "surveillance_renforcee": row.surveillance_renforcee == "Oui",
        }

        lines = [
            f"Médicament : {row.denomination} (code CIS {code})",
            f"Forme pharmaceutique : {row.forme} ; voie(s) d'administration : {row.voies.replace(';', ', ')}",
            f"Substance(s) active(s) : {'; '.join(substances) or 'non renseignée'}",
            f"Titulaire de l'AMM : {row.titulaires}",
            f"Statut : {row.statut_amm} ({row.procedure_amm}), AMM du {row.date_amm}, {row.etat_commercialisation}",
        ]
        if row.surveillance_renforcee == "Oui":
            lines.append("Médicament sous surveillance renforcée (triangle noir).")
        if row.statut_bdm:
            lines.append(f"Statut BDPM : {row.statut_bdm} (information de sécurité ou de disponibilité en cours)")
        if code in cpd.index:
            lines.append("Conditions de prescription et de délivrance : " + " ; ".join(cpd[code]))
        if code in gener.index:
            g = gener.loc[code]
            role = TYPE_GENERIQUE.get(g["type_generique"], "membre")
            lines.append(f"Groupe générique : {g['libelle_groupe']} — ce médicament en est le {role}.")
        chunks.append(Chunk(id=f"{code}:fiche", text="\n".join(lines),
                            payload={**base_payload, "type": "fiche"}))

        if code in smr_groups:
            for i, a in enumerate(smr.get_group(code).head(SMR_PER_DRUG).itertuples(index=False)):
                libelle = a.libelle_smr[:MAX_SMR_CHARS]
                text = (f"Médicament : {row.denomination} (code CIS {code})\n"
                        f"Avis de la HAS (Commission de la transparence) du {_fmt_date_has(a.date_avis)}"
                        f" — motif : {a.motif}\n"
                        f"Service médical rendu (SMR) : {a.valeur_smr}\n{libelle}")
                chunks.append(Chunk(id=f"{code}:smr:{i}", text=text,
                                    payload={**base_payload, "type": "avis_smr", "valeur_smr": a.valeur_smr}))
    return chunks


def build_vocabulary(chunks: list[Chunk]) -> list[str]:
    vocab = set()
    for c in chunks:
        vocab.update(c.payload.get("keywords", []))
    return sorted(vocab)
