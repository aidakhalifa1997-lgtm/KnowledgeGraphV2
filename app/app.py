"""OriziaBot : assistant de connaissances Orizia, interface de discussion au-dessus du Knowledge Graph.

Lancement (depuis la racine du dépôt, .venv activé, Neo4j démarré) :
    streamlit run app/app.py
"""
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

load_dotenv()
RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                      # pour importer src.agent
FICHIER_DISCUSSIONS = RACINE / "data" / "conversations.json"      # discussions récentes (local)

st.set_page_config(page_title="OriziaBot", page_icon=":material/hub:", layout="wide")

COULEURS = {"Procedure": "#C8683F", "Personne": "#C9A66B", "Acteur": "#B39A8C", "Direction": "#7F95A3",
            "Domaine": "#A8A08F", "Outil": "#B5838D", "Projet": "#8FA37E"}

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: #E9E4DA; border-right: 1px solid #DAD3C7;}
[data-testid="stSidebar"] * {color: #2F2B26;}
[data-testid="stSidebar"] .stButton button {border: none; background: transparent; box-shadow: none;
    padding: 0.45rem 0.7rem; min-height: 2.5rem; border-radius: 8px; justify-content: flex-start;}
[data-testid="stSidebar"] .stButton button div {justify-content: flex-start; text-align: left;}
[data-testid="stSidebar"] .stButton button:hover {background: #DDD6C9;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {background: #D9D2C4;}
[data-testid="stSidebar"] .stButton button p {font-size: 1.05rem !important; text-align: left; font-weight: 400;}
[data-testid="stSidebar"] .stButton button span[data-testid="stIconMaterial"] {font-size: 1.4rem; margin-right: 0.3rem;}
.section {color: #7B756A; font-size: 0.95rem; margin: 1.6rem 0 0.4rem 0.7rem;}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {font-size: 1.05rem; line-height: 1.65;}
[data-testid="stChatInput"] textarea {font-size: 1.05rem !important; min-height: 0 !important; line-height: 1.5; padding-top: 0.45rem; padding-bottom: 0.45rem;}
[data-testid="stChatInput"] > div {min-height: 0 !important; padding-top: 0.15rem !important; padding-bottom: 0.15rem !important;}
.stButton button p {white-space: normal !important;}
[data-testid="stChatInput"] {background: #F3F0EB !important; padding: 0 !important;}
[data-testid="stChatInput"] > div {background: #C8683F !important; border-color: #C8683F !important;}
[data-testid="stChatInput"] textarea {background: transparent !important; color: #FFFFFF !important;}
[data-testid="stChatInput"] textarea::placeholder {color: #F6DCCF !important; opacity: 1;}
[data-testid="stChatInput"] button {background: #FFFFFF !important; color: #C8683F !important;}
.stButton button p {font-size: 1.15rem !important;}
.stButton button {padding: 0.9rem 1rem; min-height: 3.4rem;}
[data-testid="stMarkdownContainer"] p {font-size: 1.1rem;}
[data-testid="stChatMessageAvatarAssistant"] {background: #C8683F; color: #FFFFFF;}
[data-testid="stChatMessageAvatarUser"] {background: #B39A8C;}
[data-testid="stChatMessage"] {background: transparent; padding: 0.4rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #E6E0D6; border-radius: 6px 18px 18px 18px; padding: 0.6rem 1rem;}
[data-testid="stChatInput"] {border-radius: 14px; border: 1px solid #D3CBBD; background: #FAF8F5;}
.logo {display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px;
       border-radius: 10px; background: #C8683F; color: #fff; font-weight: 700; font-size: 1.1rem; margin-right: 10px;}
.marque {font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; color: #2F2B26;}
.accueil {text-align: center; margin: 13vh 0 0 0;}
.accueil .logo {width: 64px; height: 64px; border-radius: 18px; font-size: 2rem; margin: 0 0 1rem 0;}
.accueil .titre {font-size: 2.6rem; font-weight: 700; letter-spacing: -.02em; color: #2F2B26;}
.sous-titre {text-align: center; color: #7B756A; font-size: 1.3rem; margin: .4rem 0 2.2rem 0;}
.puce {display: inline-block; background: #EAE5DB; border: 1px solid #D3CBBD; color: #4A5A50;
       border-radius: 8px; padding: 2px 10px; margin: 2px 6px 2px 0; font-size: 0.85rem;}
.source-titre {color: #7B756A; font-size: 0.75rem; margin-top: 0.8rem; letter-spacing: .08em; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ discussions récentes
def lire_discussions() -> dict:
    try:
        return json.loads(FICHIER_DISCUSSIONS.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def ecrire_discussions(discussions: dict) -> None:
    FICHIER_DISCUSSIONS.parent.mkdir(parents=True, exist_ok=True)
    FICHIER_DISCUSSIONS.write_text(json.dumps(discussions, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ accès à Neo4j
@st.cache_resource
def connexion():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")))


def requete(cypher: str, **params) -> list[dict]:
    with connexion().session() as s:
        return [r.data() for r in s.run(cypher, **params)]


def infos_procedures(references: list[str]) -> list[dict]:
    return requete("MATCH (p:Procedure) WHERE p.reference IN $refs "
                   "RETURN p.reference AS ref, p.titre AS titre ORDER BY ref", refs=references)


def voisins_de(reference: str) -> list[dict]:
    """Tout ce qui est relié directement à la procédure, avec le sens et les sources de la relation."""
    return requete(
        """
        MATCH (p:Procedure {reference: $ref})-[r]-(x)
        RETURN type(r) AS relation, startNode(r) = p AS sortant, r.sources AS sources,
               labels(x)[0] AS type,
               CASE labels(x)[0] WHEN 'Procedure' THEN x.reference WHEN 'Projet' THEN x.code ELSE x.nom END AS cle,
               coalesce(x.titre, x.description, x.libelle, '') AS detail
        """, ref=reference)


def dessiner(reference: str, titre: str, voisins: list[dict]) -> str:
    net = Network(height="560px", width="100%", directed=True, bgcolor="#FAF8F5", font_color="#2F2B26",
                  cdn_resources="in_line")
    centre = f"Procedure:{reference}"
    net.add_node(centre, label=reference, title=titre, color=COULEURS["Procedure"], size=34)
    for v in voisins:
        noeud = f"{v['type']}:{v['cle']}"
        net.add_node(noeud, label=v["cle"], title=v["detail"] or v["cle"], color=COULEURS.get(v["type"], "#999"), size=22)
        debut, fin = (centre, noeud) if v["sortant"] else (noeud, centre)
        net.add_edge(debut, fin, label=v["relation"], title="Source : " + ", ".join(v["sources"] or []),
                     font={"size": 9})
    return net.generate_html()


# ------------------------------------------------------------------ moteur de réponse (LangGraph)
def repondre(question: str) -> dict:
    """Retourne {"contenu", "sources", "explication"} en interrogeant le moteur LangGraph."""
    try:
        from src.agent.moteur import poser_question
        r = poser_question(question)
        return {"contenu": r["reponse"], "sources": r.get("sources", []), "explication": r.get("explication", "")}
    except Exception as e:                      # la démo ne doit jamais planter : on affiche l'erreur dans la réponse
        return {"contenu": f"Une erreur est survenue pendant la recherche : {type(e).__name__}. Réessayez dans un instant.",
                "sources": [], "explication": str(e)[:200]}


# ------------------------------------------------------------------ affichage
def afficher_sources(references: list[str], cle: str) -> None:
    titres = {p["ref"]: p["titre"] for p in infos_procedures(references)}
    st.markdown("<div class='source-titre'>DOCUMENTS SOURCES</div>" + "".join(
        f"<span class='puce' title='{titres.get(r, '')}'>{r} · {titres.get(r, '')}</span>" for r in references),
        unsafe_allow_html=True)
    if st.toggle("Voir le graphe", key=f"toggle_{cle}"):      # affiché seulement si demandé (rendu correct)
        choix = st.selectbox("Procédure", references, key=f"graphe_{cle}") if len(references) > 1 else references[0]
        st.markdown(" &nbsp; ".join(f"<span style='color:{c}'>●</span> {t}" for t, c in COULEURS.items()),
                    unsafe_allow_html=True)
        components.html(dessiner(choix, titres.get(choix, ""), voisins_de(choix)), height=580)


def afficher_message(m: dict, cle: str) -> None:
    with st.chat_message(m["role"], avatar=":material/hub:" if m["role"] == "assistant" else None):
        st.markdown(m["contenu"])
        if m.get("sources"):
            afficher_sources(m["sources"], cle)
        if m.get("explication"):
            st.caption("Raisonnement : " + m["explication"])


SUGGESTIONS = ["Qui est responsable de la gestion des changements ?",
               "Quelles procédures utilisent l'outil OriDesk ?",
               "Quels sont les experts de la sécurité des systèmes d'information ?",
               "Quelles procédures dépendent de la gestion des incidents ?"]

# ------------------------------------------------------------------ logique de la page
discussions = lire_discussions()
courante = st.session_state.get("discussion")

question = st.chat_input("Posez votre question sur les procédures…") or st.session_state.pop("suggestion", None)
if question:
    if courante not in discussions:
        courante = uuid.uuid4().hex[:8]
        discussions[courante] = {"titre": question[:48] + ("…" if len(question) > 48 else ""),
                                 "date": datetime.now().isoformat(), "messages": []}
    discussions[courante]["messages"].append({"role": "user", "contenu": question})
    with st.spinner("Recherche dans le graphe…"):
        discussions[courante]["messages"].append({"role": "assistant", **repondre(question)})
    discussions[courante]["date"] = datetime.now().isoformat()
    ecrire_discussions(discussions)
    st.session_state["discussion"] = courante

with st.sidebar:
    st.markdown("<span class='logo'>O</span><span class='marque'>OriziaBot</span>", unsafe_allow_html=True)
    st.write("")
    if st.button("Nouvelle discussion", icon=":material/add:", type="tertiary", use_container_width=True):
        st.session_state["discussion"] = None
        st.rerun()
    st.markdown("<div class='section'>Récentes</div>", unsafe_allow_html=True)
    for cid, d in sorted(discussions.items(), key=lambda x: x[1]["date"], reverse=True)[:15]:
        if st.button(d["titre"], key=f"d_{cid}", use_container_width=True,
                     type="secondary" if cid == courante else "tertiary"):
            st.session_state["discussion"] = cid
            st.rerun()

messages = discussions.get(courante, {}).get("messages", [])
if not messages:
    st.markdown("<div class='accueil'><span class='logo'>O</span><div class='titre'>Bonjour, je suis OriziaBot</div></div>"
                "<div class='sous-titre'>Posez votre question.</div>",
                unsafe_allow_html=True)
    colonnes = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if colonnes[i % 2].button(s, key=f"s_{i}", use_container_width=True):
            st.session_state["suggestion"] = s
            st.rerun()
for i, m in enumerate(messages):
    afficher_message(m, f"{courante}_{i}")
