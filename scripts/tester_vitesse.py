"""Mesure la vitesse de l'API d'embeddings selon la taille de la demande.
    python -m scripts.tester_vitesse                       # modèle par défaut
    python -m scripts.tester_vitesse nvidia/embed-qa-4     # un autre modèle
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
modele = sys.argv[1] if len(sys.argv) > 1 else os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"), timeout=90, max_retries=0)
passage = ("La demande d'achat est validée par le responsable de l'équipe Approvisionnement avant émission du bon de commande. " * 8)[:900]

print("Modèle :", modele)
for nb, texte in [(1, "Qui valide une demande d'achat ?"), (1, passage), (4, passage), (8, passage)]:
    debut = time.time()
    try:
        client.embeddings.create(model=modele, input=[texte] * nb, encoding_format="float",
                                 extra_body={"input_type": "passage", "truncate": "END"})
        print(f"  {nb} texte(s) de {len(texte)} caractères : {time.time() - debut:.1f} s")
    except Exception as e:
        print(f"  {nb} texte(s) de {len(texte)} caractères : ÉCHEC après {time.time() - debut:.1f} s ({type(e).__name__})")
