"""Recherche vectorielle (RAG) dans le texte des procédures.

Deux rôles :
  - embeddings() : transforme des textes en vecteurs avec le modèle NVIDIA (utilisé à l'indexation et à la question)
  - chercher()   : trouve les passages dont le vecteur est le plus proche de celui de la question
L'index (passages + vecteurs) est construit une fois par : python -m src.retrieval.indexation
"""
import json
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
RACINE = Path(__file__).resolve().parents[2]
DOSSIER_INDEX = RACINE / "data" / "index"
MODELE_EMBED = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
TOP_K = 6          # nombre de passages gardés
SEUIL = 0.30       # score minimal (cosinus) : en dessous, le passage est jugé hors sujet. À ajuster sur les tests.

POIDS_MOTS = 0.25  # poids de la correspondance de mots dans le classement (recherche hybride : sens + mots exacts)

_client = None


def mots(texte: str) -> set:
    """Mots significatifs (5 lettres ou plus, sans accents) : sert à repérer les termes exacts de la question."""
    t = unicodedata.normalize("NFD", texte.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return {m for m in re.findall(r"[a-z0-9]+", t) if len(m) >= 5}


def embeddings(textes: list[str], type_entree: str, afficher: bool = False) -> np.ndarray:
    """type_entree : "passage" (texte indexé) ou "query" (question). Retourne des vecteurs normalisés."""
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"),
                         timeout=45, max_retries=0)          # un appel bloqué est coupé au bout de 45 s, puis on réessaie
    vecteurs = []
    nb_lots = (len(textes) + 7) // 8
    for i in range(0, len(textes), 8):                       # petits lots pour ne pas dépasser les limites de l'API
        lot = textes[i:i + 8]
        if afficher:
            print(f"  lot {i // 8 + 1}/{nb_lots}", flush=True)
        for essai in range(4):
            try:
                r = _client.embeddings.create(model=MODELE_EMBED, input=lot, encoding_format="float",
                                              extra_body={"input_type": type_entree, "truncate": "END"})
                break
            except Exception as e:
                if afficher:
                    print(f"    échec ({type(e).__name__}), nouvel essai…", flush=True)
                if essai == 3:
                    raise
                time.sleep(2 * (essai + 1))
        vecteurs += [d.embedding for d in r.data]
    m = np.array(vecteurs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def charger_index():
    fichier_passages, fichier_vecteurs = DOSSIER_INDEX / "passages.json", DOSSIER_INDEX / "vecteurs.npy"
    if not (fichier_passages.exists() and fichier_vecteurs.exists()):
        return None
    passages = json.loads(fichier_passages.read_text(encoding="utf-8"))
    return passages, np.load(fichier_vecteurs), [mots(p["texte"]) for p in passages]


def chercher(question: str, top_k: int = TOP_K, seuil: float = SEUIL, references: set | None = None) -> list[dict]:
    """Retourne [{"reference", "texte", "score"}] des passages les plus pertinents, du meilleur au moins bon.
    Classement hybride : proximité de sens (vecteurs) + présence des mots de la question (termes exacts).
    references : si fourni, la recherche est limitée à ces procédures."""
    index = charger_index()
    if index is None:
        return []
    passages, matrice, mots_passages = index
    cosinus = matrice @ embeddings([question], "query")[0]
    mq = mots(question)
    recouvrement = np.array([len(mq & m) / len(mq) if mq else 0.0 for m in mots_passages])
    final = cosinus + POIDS_MOTS * recouvrement
    if references:
        final = np.where([p["reference"] in references for p in passages], final, -1.0)
    ordre = [i for i in np.argsort(-final)[:top_k] if final[i] > -1 and cosinus[i] >= seuil]
    return [{"reference": passages[i]["reference"], "texte": passages[i]["texte"], "score": round(float(cosinus[i]), 3)}
            for i in ordre]
