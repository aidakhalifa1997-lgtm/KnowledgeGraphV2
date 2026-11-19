"""Teste le nœud graphe sans rédaction : affiche ce que le graphe renvoie pour quelques questions.
    python -m scripts.tester_graphe                 # questions prédéfinies
    python -m scripts.tester_graphe "Votre question ?"
"""
import json
import sys

from src.agent.graphe import chercher
from src.agent.moteur import appeler_llm

QUESTIONS = [
    "Qui est responsable de la gestion des changements ?",
    "Quelles procédures utilisent l'outil OriDesk ?",
    "Quels sont les experts de la sécurité des systèmes d'information ?",
    "Quelles procédures dépendent de la gestion des incidents ?",
    "Quelles procédures sont proches de ACH-01 ?",
    "Quels contrôles prévoit la procédure ACH-01 ?",
]

for q in (sys.argv[1:] or QUESTIONS):
    r = chercher(q, appeler_llm)
    print("\n" + "=" * 90)
    print("Question  :", q)
    print("Analyse   :", r["intention"] or "?", "|", r["mention"] or "?")
    print("Nœud relié:", [f"{n['type']}:{n['cle']} ({n['score']})" for n in r["noeuds"]] or "aucun")
    print("Résultats :", len(r["lignes"]), "ligne(s)", ("- " + r["diagnostic"]) if r["diagnostic"] else "")
    for l in r["lignes"][:6]:
        print("   ", json.dumps(l, ensure_ascii=False))
    print("Sources   :", r["sources"])
