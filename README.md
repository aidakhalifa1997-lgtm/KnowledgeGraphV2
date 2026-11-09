# Plateforme de gestion des connaissances : Knowledge Graph et LLM

Projet de fin d'études du Master Data Engineering (EHTP).
Plateforme qui extrait les connaissances de documents d'entreprise, construit un graphe de connaissances dans Neo4j et répond à des questions en langage naturel en s'appuyant sur ce graphe (GraphRAG).

## État d'avancement

- [x] Squelette du dépôt et Neo4j sous Docker
- [x] Lecture des documents Word, PDF natifs et PDF scannés (OCR)
- [ ] Extraction des entités et des relations par le LLM
- [ ] Chargement dans Neo4j et recherche vectorielle
- [ ] Recherche par le graphe et routage (LangGraph)
- [ ] Évaluation et interface

## Données

Corpus fictif de 30 procédures d'une banque fictive, **Orizia**, couvrant cinq domaines (IT, sécurité IT, achats et logistique, finance, RH). Les personnes, outils, projets, seuils et délais sont inventés.

À copier dans le dépôt (non fournis par Git, voir la section suivante) :

| Dossier | Contenu |
|---|---|
| `data/raw/word/` | procédures au format Word |
| `data/raw/pdf_natif/` | procédures au format PDF avec texte |
| `data/raw/scans/` | procédures scannées (image seule), pour l'OCR |
| `data/reference/` | relations de référence et registre, pour l'évaluation |

## Prérequis

- Docker Desktop
- Python 3.11 ou plus récent
- Tesseract avec le pack de langue française (`tesseract --list-langs` doit afficher `fra`)

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Mac/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

Copier `.env.example` en `.env` puis renseigner `NEO4J_PASSWORD` (8 caractères minimum). Le fichier `.env` n'est jamais versionné.

## Lancer Neo4j

```bash
docker compose up -d
```

Interface web : http://localhost:7474 (utilisateur `neo4j`, mot de passe du `.env`).
Vérifier la connexion depuis Python : `python scripts/verifier_neo4j.py`.

## Lire les documents

```bash
python -m src.ingestion.lire_documents
```

Le texte extrait est écrit dans `data/extracted/`.

## Structure

```
data/            corpus (raw), références d'évaluation, texte extrait
src/ingestion/   lecture des documents et OCR
src/graph/       construction du graphe dans Neo4j
src/retrieval/   recherche vectorielle et recherche par le graphe
src/agent/       routage et orchestration (LangGraph)
src/evaluation/  banc d'évaluation
scripts/         scripts utilitaires
tests/           tests automatiques
```
