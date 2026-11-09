"""Vérifie que Python arrive à se connecter à Neo4j. Usage : python scripts/verifier_neo4j.py"""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
utilisateur = os.getenv("NEO4J_USER", "neo4j")
mot_de_passe = os.getenv("NEO4J_PASSWORD")

if not mot_de_passe:
    raise SystemExit("NEO4J_PASSWORD est vide : vérifiez votre fichier .env")

with GraphDatabase.driver(uri, auth=(utilisateur, mot_de_passe)) as driver:
    driver.verify_connectivity()
    with driver.session() as session:
        valeur = session.run("RETURN 1 AS ok").single()["ok"]
print("Connexion à Neo4j OK" if valeur == 1 else "Réponse inattendue")
