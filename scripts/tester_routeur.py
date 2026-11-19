"""Teste le routeur sur des questions types (racine du dépôt, .venv activé) :
    python -m scripts.tester_routeur
"""
from src.agent.moteur import poser_question

QUESTIONS = [
    ("Qui est responsable de la gestion des changements ?", "relations"),
    ("Quelles procédures utilisent l'outil OriDesk ?", "relations"),
    ("Quels sont les experts de la sécurité des systèmes d'information ?", "relations"),
    ("Quelles procédures dépendent de la gestion des incidents ?", "relations"),
    ("Quel est le délai de validation d'une demande d'achat ?", "contenu"),
    ("Que dit la procédure ACH-01 sur les étapes de la demande d'achat ?", "contenu"),
    ("Comment déclarer un incident de sécurité ?", "contenu"),
    ("Quelles procédures de la direction Achats mentionnent une double validation ?", "mixte"),
]

bons = 0
for question, attendu in QUESTIONS:
    obtenu = poser_question(question)["type"]
    bons += obtenu == attendu
    print(f"{'OK ' if obtenu == attendu else 'KO '} attendu={attendu:<9} obtenu={obtenu:<9} | {question}")
print(f"\nRoutage correct : {bons}/{len(QUESTIONS)}")
