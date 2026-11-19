"""Moteur de questions-réponses OriziaBot, construit comme un graphe LangGraph.

Chemin d'une question :
    routeur  ->  (graphe | vectoriel | graphe puis vectoriel)  ->  repondre

 - routeur    : le LLM classe la question : "contenu", "relations" ou "mixte"
 - graphe     : recherche dans Neo4j (GraphRAG)                [src/agent/graphe.py]
 - vectoriel  : recherche dans le texte des procédures (RAG)    [src/retrieval/vectoriel.py]
 - repondre   : rédaction de la réponse avec les sources       [à écrire ensuite]

Ce fichier contient pour l'instant le squelette et le routeur ; les trois autres nœuds sont des emplacements.
"""
import json
import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.agent import graphe
from src.retrieval import vectoriel

load_dotenv()

MODELE = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
TYPES = ("contenu", "relations", "mixte")


class Etat(TypedDict, total=False):
    """Ce que les nœuds se transmettent."""
    question: str
    type: str                 # décidé par le routeur
    resultats_graphe: dict    # rempli par le nœud graphe (intention, noeuds, lignes, sources, diagnostic)
    passages: list            # rempli par le nœud vectoriel : [{reference, texte, score}]
    reponse: str
    sources: list             # références de procédures citées
    explication: str          # comment la réponse a été trouvée


# ------------------------------------------------------------------ appel du LLM
_client = None


def appeler_llm(consigne: str, question: str, max_tokens: int = 200) -> str:
    """Un appel au LLM, sans mode « réflexion » pour rester rapide."""
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    r = _client.chat.completions.create(
        model=MODELE, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": consigne}, {"role": "user", "content": question}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""


# ------------------------------------------------------------------ nœud 1 : le routeur
CONSIGNE_ROUTEUR = """Tu classes des questions posées sur une base de procédures bancaires.
Réponds par UN SEUL mot parmi : contenu, relations, mixte.

- contenu : la réponse se trouve dans le TEXTE d'une procédure (règle, étape, délai, seuil, définition, « que dit... », « comment... »).
- relations : la réponse vient des LIENS entre éléments (qui est responsable, quel expert, quels outils, quelles procédures dépendent d'une autre, procédures similaires, quelle direction).
- mixte : la question demande à la fois des liens et le contenu du texte.

Exemples :
Qui est responsable de la gestion des changements ? -> relations
Quelles procédures utilisent l'outil OriDesk ? -> relations
Quel est le délai de validation d'une demande d'achat ? -> contenu
Quelles procédures de la direction Achats mentionnent une double validation ? -> mixte"""


def decider_type(reponse_llm: str) -> str:
    """Lit la réponse du LLM. En cas de doute, 'mixte' : on cherche partout plutôt que de se tromper."""
    mot = re.sub(r"[^a-zé]", "", reponse_llm.lower().split()[0]) if reponse_llm.split() else ""
    return mot if mot in TYPES else "mixte"


def routeur(etat: Etat) -> Etat:
    return {"type": decider_type(appeler_llm(CONSIGNE_ROUTEUR, etat["question"], max_tokens=10))}


def choisir_branche(etat: Etat) -> str:
    """Utilisé par LangGraph pour décider du nœud suivant."""
    return {"relations": "graphe", "contenu": "vectoriel", "mixte": "graphe"}[etat["type"]]


def apres_graphe(etat: Etat) -> str:
    """Question mixte, ou graphe sans résultat (repli) : on cherche aussi dans le texte."""
    g = etat["resultats_graphe"]
    incomplet = not g["lignes"] or bool(g.get("remarque", "").strip())      # rien trouvé, ou information manquante dans le graphe
    return "vectoriel" if etat["type"] == "mixte" or incomplet else "repondre"


# ------------------------------------------------------------------ nœuds à écrire (emplacements)
def noeud_graphe(etat: Etat) -> Etat:
    return {"resultats_graphe": graphe.chercher(etat["question"], appeler_llm)}


def noeud_vectoriel(etat: Etat) -> Etat:
    try:
        g = etat.get("resultats_graphe") or {}
        citees = set(re.findall(r"\b[A-Z]{2,3}-\d{2}\b", etat["question"].upper()))
        if citees:                                         # procédure nommée dans la question : on cherche dedans
            perimetre = citees
        elif g.get("lignes") and (etat["type"] == "mixte" or g.get("remarque", "").strip()):
            perimetre = set(g.get("sources", []))          # le graphe a ciblé le périmètre (ou n'a pas tout : on complète)
        else:
            perimetre = None
        return {"passages": vectoriel.chercher(etat["question"], references=perimetre)}
    except Exception:                       # API d'embeddings indisponible : on continue avec le graphe seul
        return {"passages": []}


CONSIGNE_REPONSE = """Tu es OriziaBot, l'assistant des procédures de la banque Orizia. Tu réponds en français, de façon claire et professionnelle.

Règles :
- Commence par la réponse directe en une phrase, puis donne le détail si nécessaire.
- Utilise UNIQUEMENT les faits du graphe et les passages de procédures fournis. N'ajoute rien venant d'ailleurs.
- Cite la procédure source entre crochets, par exemple [IT-02].
- Si une REMARQUE est fournie, signale-la en premier avec tes mots, sans écrire « Non applicable », puis présente quand même les autres faits. N'affirme jamais qu'un rôle existe si la remarque dit le contraire.
- Quand les faits contiennent des personnes (« personne ») et des équipes (« equipe »), présente d'abord les personnes avec leur rôle, puis les équipes. Ne supprime aucune personne.
- Si les faits concernent plusieurs éléments (colonne « element »), signale que la question est ambiguë et présente chaque cas séparément.
- N'écris jamais de noms techniques (A_POUR_..., relation, nœud). Ne commente pas la formulation de la question.
- Si la REMARQUE dit qu'une information manque dans le graphe, cherche-la dans les passages : si elle y figure, donne-la en citant la procédure ; sinon dis qu'elle n'a pas été trouvée.
- Quand on te demande quelles procédures « mentionnent » ou « prévoient » quelque chose, ne cite que celles dont un passage contient explicitement cette information. Si aucun passage ne la contient, réponds qu'aucune procédure du périmètre ne la mentionne. N'invente jamais.
- Réponds précisément à la question posée : ne confonds pas deux délais, deux seuils ou deux étapes différents. Si les passages ne donnent pas exactement l'information demandée, dis-le.
- Cite la procédure entre crochets à chaque information chiffrée.
- Pour des procédures « proches » ou « similaires », indique ce qu'elles ont en commun (colonne « en_commun »).
- Réponse courte (liste seulement s'il y a plusieurs éléments)."""

MAX_FAITS = 40


def faits_en_texte(g: dict) -> str:
    entete = f"REMARQUE : {g['remarque'].strip()}\n" if g.get("remarque", "").strip() else ""
    return entete + "\n".join(json.dumps(l, ensure_ascii=False) for l in g["lignes"][:MAX_FAITS])


def explication(g: dict, passages: list, type_question: str) -> str:
    """Comment la réponse a été trouvée (affichée sous la réponse)."""
    morceaux = [f"Question de type « {type_question} »"]
    if g:
        noeuds = ", ".join(f"{n['type']} {n['cle']}" for n in g["noeuds"]) or "aucun élément reconnu"
        morceaux.append(f"graphe : intention « {g['intention'] or '?'} », {noeuds}, {len(g['lignes'])} résultat(s)")
    if passages:
        morceaux.append(f"texte des procédures : {len(passages)} passage(s) (meilleur score {passages[0]['score']})")
    return " · ".join(morceaux)



def repondre(etat: Etat) -> Etat:
    g = etat.get("resultats_graphe") or {}
    passages = etat.get("passages") or []
    expl = explication(g, passages, etat["type"])
    if not g.get("lignes") and not passages:      # rien trouvé nulle part : on ne laisse pas le LLM inventer
        motif = g.get("diagnostic") or "aucun passage pertinent"
        return {"reponse": f"Je n'ai pas trouvé d'élément permettant de répondre ({motif}). "
                           "Pouvez-vous reformuler ou préciser la procédure, l'outil ou le domaine concerné ?",
                "sources": [], "explication": expl}
    contexte = f"Question : {etat['question']}"
    if g.get("lignes"):
        contexte += f"\n\nFaits issus du graphe :\n{faits_en_texte(g)}"
    if passages:
        contexte += "\n\nPassages des procédures :\n" + "\n\n".join(f"[{p['reference']}] {p['texte']}" for p in passages)
    texte = appeler_llm(CONSIGNE_REPONSE, contexte, max_tokens=900)
    texte = re.sub(r"^\s*Non applicable\.?\s*", "", texte, flags=re.IGNORECASE)    # tournure parasite
    disponibles = set(g.get("sources", [])) | {p["reference"] for p in passages}
    citees = [r for r in dict.fromkeys(re.findall(r"\b[A-Z]{2,3}-\d{2}\b", texte)) if r in disponibles]
    return {"reponse": texte.strip(), "sources": citees or sorted(disponibles)[:8], "explication": expl}


# ------------------------------------------------------------------ assemblage du graphe LangGraph
def construire():
    g = StateGraph(Etat)
    g.add_node("routeur", routeur)
    g.add_node("graphe", noeud_graphe)
    g.add_node("vectoriel", noeud_vectoriel)
    g.add_node("repondre", repondre)
    g.set_entry_point("routeur")
    g.add_conditional_edges("routeur", choisir_branche, {"graphe": "graphe", "vectoriel": "vectoriel"})
    g.add_conditional_edges("graphe", apres_graphe, {"vectoriel": "vectoriel", "repondre": "repondre"})
    g.add_edge("vectoriel", "repondre")
    g.add_edge("repondre", END)
    return g.compile()


moteur = construire()


def poser_question(question: str) -> dict:
    """Point d'entrée utilisé par l'application."""
    return moteur.invoke({"question": question})
