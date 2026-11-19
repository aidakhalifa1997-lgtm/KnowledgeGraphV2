"""Construit l'index du RAG : découpe les textes des procédures en passages et calcule leurs vecteurs.

Usage (racine du dépôt, .venv activé) :   python -m src.retrieval.indexation
Entrée : data/extracted/<reference>_*.txt        Sortie : data/index/passages.json et data/index/vecteurs.npy
À relancer si les textes extraits changent.
"""
import csv
import json
import re

import numpy as np

from src.retrieval.vectoriel import DOSSIER_INDEX, RACINE, embeddings

DOSSIER_TEXTES = RACINE / "data" / "extracted"
FICHIER_TITRES = RACINE / "data" / "referentiel" / "referentiel_procedures.csv"
TAILLE_MAX = 900      # caractères par passage (environ un court paragraphe à une section)


def lire_titres() -> dict:
    try:
        with open(FICHIER_TITRES, encoding="utf-8-sig", newline="") as f:
            return {r["reference"].strip().upper(): r["titre"].strip() for r in csv.DictReader(f, delimiter=";")}
    except FileNotFoundError:
        return {}


def decouper(texte: str) -> list[str]:
    """Regroupe les paragraphes jusqu'à TAILLE_MAX caractères ; un paragraphe trop long est coupé par phrases."""
    morceaux = []
    for para in re.split(r"\n\s*\n", texte):
        para = re.sub(r"[ \t]+", " ", para).strip()
        while len(para) > TAILLE_MAX:
            coupe = para.rfind(". ", 0, TAILLE_MAX)
            coupe = coupe + 1 if coupe > 200 else TAILLE_MAX
            morceaux.append(para[:coupe].strip())
            para = para[coupe:].strip()
        if para:
            morceaux.append(para)
    passages, courant = [], ""
    for m in morceaux:
        if courant and len(courant) + len(m) > TAILLE_MAX:
            passages.append(courant)
            courant = ""
        courant = (courant + "\n" + m).strip()
    if courant:
        passages.append(courant)
    return [p for p in passages if len(p) > 40]


def main():
    titres = lire_titres()
    passages = []
    for fichier in sorted(DOSSIER_TEXTES.glob("*.txt")):
        m = re.match(r"^([A-Za-z]{2,3}-\d{2})", fichier.name)
        if not m:
            continue
        ref = m.group(1).upper()
        for p in decouper(fichier.read_text(encoding="utf-8", errors="ignore")):
            passages.append({"reference": ref, "texte": p})
    print(f"{len(passages)} passages issus de {len({p['reference'] for p in passages})} procédures. Calcul des vecteurs…")
    # on ajoute la référence et le titre devant chaque passage pour que le vecteur « sache » de quelle procédure il vient
    a_indexer = [f"[{p['reference']} – {titres.get(p['reference'], '')}]\n{p['texte']}" for p in passages]
    vecteurs = embeddings(a_indexer, "passage", afficher=True)
    DOSSIER_INDEX.mkdir(parents=True, exist_ok=True)
    (DOSSIER_INDEX / "passages.json").write_text(json.dumps(passages, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(DOSSIER_INDEX / "vecteurs.npy", vecteurs)
    print(f"Index enregistré dans {DOSSIER_INDEX} (dimension {vecteurs.shape[1]}).")


if __name__ == "__main__":
    main()
