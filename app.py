"""Interface de démonstration : streamlit run app.py"""
from __future__ import annotations

import re

import streamlit as st

from bdpm_rag.config import settings
from bdpm_rag.graph import default_pipeline

st.set_page_config(page_title="BDPM RAG", page_icon="💊", layout="wide")


@st.cache_resource
def pipeline():
    return default_pipeline()


st.title("💊 Assistant médicament — BDPM")
st.caption(
    f"RAG local sur la Base de Données Publique des Médicaments (ANSM/HAS) · "
    f"LLM `{settings.llm_model}` · embeddings `{settings.embed_model}` via Ollama · Qdrant · LangGraph"
)
st.warning("Démonstrateur technique : ne remplace pas le RCP ni l'avis d'un professionnel de santé.", icon="⚠️")

with st.sidebar:
    st.subheader("Exemples")
    examples = [
        "Quelles sont les conditions de prescription d'Ozempic ?",
        "Quel est le service médical rendu de Xarelto ?",
        "Levothyrox est-il un princeps ou un générique ?",
        "Quels médicaments contiennent de l'apixaban ?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

question = st.chat_input("Posez une question sur un médicament…") or st.session_state.pop("question", None)

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la BDPM…"):
            result = pipeline().invoke({"question": question})
        answer = result["answer"]
        # Rend les citations [n] cliquables vers la fiche BDPM correspondante
        hits = result.get("hits", [])

        def link(m: re.Match) -> str:
            i = int(m.group(1))
            return f"[[{i}]]({hits[i - 1].payload['url']})" if 1 <= i <= len(hits) else m.group(0)

        st.markdown(re.sub(r"\[(\d+)\]", link, answer))
        for w in result.get("warnings", []):
            st.warning(w)
        if result.get("keywords"):
            st.caption("Entités reconnues : " + ", ".join(result["keywords"]))

    if hits:
        st.subheader("Sources")
        for i, h in enumerate(hits, start=1):
            p = h.payload
            label = "Avis SMR HAS" if p["type"] == "avis_smr" else "Fiche"
            with st.expander(f"[{i}] {p['denomination']} — {label} · score {h.score:.3f}"):
                st.text(h.text)
                st.markdown(f"[Voir la fiche officielle BDPM]({p['url']}) · CIS `{p['cis']}`")
