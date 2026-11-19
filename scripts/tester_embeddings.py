"""Test des embeddings NVIDIA sur du texte français.

Usage (racine du dépôt, .venv activé) :
    python scripts/tester_embeddings.py                      # liste les modèles d'embeddings disponibles
    python scripts/tester_embeddings.py NOM_DU_MODELE        # teste ce modèle
"""
import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))

# 1) Sans argument : on affiche les modèles qui ressemblent à des modèles d'embeddings
if len(sys.argv) < 2:
    noms = sorted(m.id for m in client.models.list().data)
    candidats = [n for n in noms if "embed" in n.lower() or "retriev" in n.lower()]
    print("Modèles d'embeddings trouvés :")
    for n in candidats:
        print("  -", n)
    print("\nRelance avec l'un d'eux : python scripts/tester_embeddings.py NOM_DU_MODELE")
    sys.exit()

modele = sys.argv[1]
question = "Qui valide une demande d'achat ?"
passages = [
    "La demande d'achat est validée par le responsable de l'équipe Approvisionnement avant émission du bon de commande.",
    "La sauvegarde des serveurs est réalisée chaque nuit par l'équipe Infrastructure.",
    "Le recrutement d'un nouveau collaborateur nécessite l'accord de la Direction des Ressources Humaines.",
]


def vecteurs(textes, type_entree):
    """Appelle l'API. Certains modèles exigent input_type (query ou passage) : on essaie avec, puis sans."""
    try:
        r = client.embeddings.create(model=modele, input=textes, encoding_format="float",
                                     extra_body={"input_type": type_entree, "truncate": "END"})
    except Exception as e:
        print(f"(avec input_type : {type(e).__name__}, nouvel essai sans)")
        r = client.embeddings.create(model=modele, input=textes, encoding_format="float")
    return [d.embedding for d in r.data]


def cosinus(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


vq = vecteurs([question], "query")[0]
vp = vecteurs(passages, "passage")
print(f"Modèle : {modele} | dimension des vecteurs : {len(vq)}\n")
print(f"Question : {question}\n")
scores = [cosinus(vq, v) for v in vp]
for p, s in zip(passages, scores):
    print(f"  {s:.3f}  {p[:80]}")
ok = scores[0] == max(scores)
print("\nRésultat :", "OK, le passage pertinent est le mieux classé." if ok else "ATTENTION, le passage pertinent n'est pas en tête.")
