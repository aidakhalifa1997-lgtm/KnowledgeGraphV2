"""Test de la clé NVIDIA : liste les modèles Nemotron accessibles."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # lit NVIDIA_API_KEY dans le fichier .env

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

for modele in client.models.list():
    if "nemotron" in modele.id.lower():
        print(modele.id)
        