"""Test d'un appel réel au LLM NVIDIA.
Usage : python scripts\\tester_chat.py [identifiant_du_modèle]
"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

modele = sys.argv[1] if len(sys.argv) > 1 else "nvidia/nemotron-3.5-lightning-30b-a3b"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=120,
)

reponse = client.chat.completions.create(
    model=modele,
    messages=[{
        "role": "user",
        "content": "Réponds en une phrase, en français : à quoi sert une procédure dans une banque ?",
    }],
    temperature=0,
    max_tokens=500,
)

choix = reponse.choices[0]
print("Modèle :", modele)
print("Réponse :", choix.message.content)
print("Fin de génération :", choix.finish_reason)