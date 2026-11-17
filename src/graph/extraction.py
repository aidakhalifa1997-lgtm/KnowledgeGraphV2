"""Extraction des entités et relations d'une procédure avec le LLM NVIDIA.

Usage (depuis la racine du dépôt, avec le .venv activé) :
    python -m src.graph.extraction IT-02     # une seule procédure
    python -m src.graph.extraction           # toutes les procédures (celles déjà traitées sont ignorées)
    python -m src.graph.extraction --refaire # retraite toutes les procédures (après un changement de prompt)

Entrée  : data/extracted/<reference>_*.txt   (texte produit par lire_documents)
Sortie  : data/extracted/json/<reference>.json
"""
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_TEXTES = RACINE / "data" / "extracted"
DOSSIER_JSON = DOSSIER_TEXTES / "json"
FICHIER_ALIAS = RACINE / "data" / "referentiel" / "referentiel_alias.csv"
MODELE = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
MAX_TOKENS = int(os.getenv("NVIDIA_MAX_TOKENS", "8000"))
# Les modèles "à réflexion" écrivent leur raisonnement avant la réponse, ce qui consomme
# les tokens. Pour l'extraction on désactive la réflexion (NVIDIA_REFLEXION=oui pour l'activer).
REFLEXION = os.getenv("NVIDIA_REFLEXION", "non").lower() == "oui"

# Relations autorisées et leur définition (schéma : docs/schema_graphe.md)
RELATIONS = {
    "A_POUR_PROPRIETAIRE": "la personne propriétaire de la procédure",
    "A_POUR_REDACTEUR": "la personne qui rédige la procédure (rédacteur)",
    "A_POUR_CONTRIBUTEUR": "une personne qui contribue, ou qui vérifie la procédure (vérificateur)",
    "A_POUR_VALIDEUR": "une personne qui approuve la procédure (approbateur)",
    "RATTACHEE_A_DIRECTION": "la direction à laquelle la procédure est rattachée",
    "APPARTIENT_AU_DOMAINE": "le domaine de la procédure",
    "RATTACHEE_AU_PROJET": "le code du projet cité par la procédure",
    "UTILISE_OUTIL": "un outil ou une application utilisé par la procédure",
    "IMPLIQUE_ACTEUR": "un acteur (poste, équipe, comité ou profil) qui intervient dans la procédure",
    "PREVOIT_CONTROLE_PAR": "un acteur chargé d'un contrôle prévu par la procédure",
    "ALIMENTE": "une procédure liée marquée « Procédure aval »",
    "DEPEND_DE": "une procédure liée marquée « Procédure amont »",
    "COMPLETE": "une procédure liée marquée « Procédure complémentaire »",
    "OCCUPE_FONCTION": "sujet = une personne, objet = sa fonction (poste)",
    "MEMBRE_DE": "sujet = une personne, objet = sa direction d'appartenance",
}
NATURES = ["poste", "équipe", "comité", "profil"]
TYPES_FERMES = ["Direction", "Domaine", "Outil", "Projet"]


def charger_noms_officiels() -> dict:
    """Lit le référentiel et retourne {type: [noms officiels]} pour les types fermés."""
    noms = {t: [] for t in TYPES_FERMES}
    with open(FICHIER_ALIAS, encoding="utf-8", newline="") as f:
        for ligne in csv.DictReader(f, delimiter=";"):
            t, nom = ligne["type"], ligne["nom_officiel"]
            if t in noms and nom not in noms[t]:
                noms[t].append(nom)
    return noms


def construire_prompt(reference: str, noms_officiels: dict) -> str:
    relations = "\n".join(f"- {r} : {d}" for r, d in RELATIONS.items())
    listes = "\n".join(f"- {t} : {', '.join(n)}" for t, n in noms_officiels.items())
    return f"""Tu analyses une procédure bancaire pour construire un graphe de connaissances.
La procédure analysée a pour référence : {reference}.
Le texte est fourni tel quel ; dans les tableaux, les cellules sont séparées par " | ".

Extrais uniquement les relations suivantes (aucune autre) :
{relations}

Règles :
1. Pour les relations de procédure, le sujet est "{reference}". Pour OCCUPE_FONCTION et MEMBRE_DE, le sujet est le nom de la personne.
2. Pour ALIMENTE, DEPEND_DE et COMPLETE, l'objet est la référence de la procédure liée (ex. IT-05), d'après la section des procédures liées.
3. N'extrais que ce qui est écrit dans le texte. N'invente rien. Si une information est absente, ne produis pas la relation.
4. Pour les Direction, Domaine, Outil et Projet, utilise exactement un nom officiel de cette liste :
{listes}
5. Les noms de personnes s'écrivent sans fonction entre parenthèses (ex. "Nadia Berrada").
6. Pour chaque acteur cité (objets de IMPLIQUE_ACTEUR, PREVOIT_CONTROLE_PAR et OCCUPE_FONCTION), indique sa nature : {', '.join(NATURES)}. La nature se donne uniquement dans la liste "acteurs", jamais comme une relation.
7. PREVOIT_CONTROLE_PAR : uniquement les acteurs de la colonne « Responsable du contrôle » du tableau des contrôles.
8. Liens entre procédures : uniquement ceux du tableau des procédures liées, jamais ceux cités dans le texte courant. Le type du lien se lit dans son libellé : « Procédure amont » = DEPEND_DE, « Procédure aval » = ALIMENTE, « Procédure complémentaire » = COMPLETE. Une procédure liée donne une seule relation.
9. Le propriétaire de la procédure figure en tête du tableau des vérificateurs : ne le compte pas comme contributeur.

Réponds uniquement avec un objet JSON, sans texte autour, de cette forme :
{{"relations": [{{"sujet": "...", "relation": "...", "objet": "..."}}],
 "acteurs": [{{"nom": "...", "nature": "..."}}]}}"""


def analyser_reponse(texte: str) -> dict:
    """Retrouve le JSON dans la réponse du modèle, même s'il y a du texte autour
    ou si le modèle a écrit un objet JSON par ligne."""
    decodeur = json.JSONDecoder()
    objets, i = [], 0
    while True:
        i = texte.find("{", i)
        if i == -1:
            break
        try:
            objet, fin = decodeur.raw_decode(texte, i)
            objets.append(objet)
            i = fin
        except json.JSONDecodeError:
            i += 1
    objets = [o for o in objets if isinstance(o, dict)]
    # cas normal : un objet complet {"relations": [...], "acteurs": [...]} (on garde le dernier)
    complets = [o for o in objets if "relations" in o]
    if complets:
        return complets[-1]
    # cas alternatif : un objet JSON par relation et par acteur
    relations = [o for o in objets if "relation" in o]
    acteurs = [o for o in objets if "nature" in o and "nom" in o]
    if relations or acteurs:
        return {"relations": relations, "acteurs": acteurs}
    raise ValueError("Aucun JSON exploitable dans la réponse du modèle")


RELATIONS_PERSONNES = ("OCCUPE_FONCTION", "MEMBRE_DE")
RELATIONS_ENTRE_PROCEDURES = ("ALIMENTE", "DEPEND_DE", "COMPLETE")
MODELE_REFERENCE = re.compile(r"^[A-Z]{2,3}-\d{2}$")  # ex. IT-05, SEC-04


def valider(donnees: dict, reference: str) -> tuple[list, list]:
    """Sépare les relations valides des rejetées (type inconnu, champ vide, sujet ou objet incohérent)."""
    valides, rejetees = [], []
    for r in donnees.get("relations", []):
        if not isinstance(r, dict):
            rejetees.append(r)
            continue
        complet = all(str(r.get(c, "")).strip() for c in ("sujet", "relation", "objet"))
        ok = complet and r["relation"] in RELATIONS
        if ok and r["relation"] not in RELATIONS_PERSONNES:
            ok = str(r["sujet"]).strip() == reference      # le sujet est la procédure analysée
        if ok and r["relation"] in RELATIONS_ENTRE_PROCEDURES:
            ok = bool(MODELE_REFERENCE.match(str(r["objet"]).strip())) and str(r["objet"]).strip() != reference
        if ok:
            valides.append({c: str(r[c]).strip() for c in ("sujet", "relation", "objet")})
        else:
            rejetees.append(r)
    return valides, rejetees


def extraire(texte: str, reference: str, client: OpenAI, noms_officiels: dict) -> dict:
    reponse = client.chat.completions.create(
        model=MODELE,
        messages=[
            {"role": "system", "content": construire_prompt(reference, noms_officiels)},
            {"role": "user", "content": texte},
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": REFLEXION}},
    )
    choix = reponse.choices[0]
    contenu = choix.message.content or ""
    DOSSIER_JSON.mkdir(parents=True, exist_ok=True)
    (DOSSIER_JSON / f"{reference}_brut.txt").write_text(contenu, encoding="utf-8")  # pour diagnostic
    if choix.finish_reason == "length":
        raise ValueError(f"Réponse coupée : limite de {MAX_TOKENS} tokens atteinte avant la fin du JSON")
    if not contenu.strip():
        raise ValueError(f"Réponse vide du modèle (finish_reason = {choix.finish_reason})")
    donnees = analyser_reponse(contenu)
    valides, rejetees = valider(donnees, reference)
    return {
        "reference": reference,
        "modele": MODELE,
        "relations": valides,
        "acteurs": donnees.get("acteurs", []),
        "rejetees": rejetees,
    }


def main(cibles: list[str]) -> None:
    DOSSIER_JSON.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=300,
    )
    noms_officiels = charger_noms_officiels()
    refaire = "--refaire" in cibles          # retraite aussi les procédures déjà extraites
    cibles = [c for c in cibles if not c.startswith("--")]
    fichiers = sorted(DOSSIER_TEXTES.glob("*.txt"))
    if cibles:
        fichiers = [f for f in fichiers if f.stem.split("_")[0] in cibles]
    if not fichiers:
        print("Aucun texte trouvé dans data/extracted (lancez d'abord lire_documents).")
        return
    for f in fichiers:
        reference = f.stem.split("_")[0]
        sortie = DOSSIER_JSON / f"{reference}.json"
        if sortie.exists() and not cibles and not refaire:
            print(f"{reference} : déjà traité, ignoré")
            continue
        try:
            resultat = extraire(f.read_text(encoding="utf-8"), reference, client, noms_officiels)
        except Exception as erreur:
            print(f"ERREUR {reference} : {erreur}")
            traceback.print_exc()
            continue
        sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{reference} : {len(resultat['relations'])} relations, "
              f"{len(resultat['rejetees'])} rejetées")
        time.sleep(2)  # reste sous la limite de requêtes par minute


if __name__ == "__main__":
    main(sys.argv[1:])
