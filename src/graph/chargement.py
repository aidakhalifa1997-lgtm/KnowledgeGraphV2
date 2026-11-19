"""Charge les JSON d'extraction dans Neo4j.

Usage (depuis la racine du dépôt, .venv activé, Neo4j démarré) :
    python -m src.graph.chargement --simulation   # prépare tout, affiche le bilan, n'écrit rien
    python -m src.graph.chargement                # charge dans Neo4j (relançable sans doublons)
    python -m src.graph.chargement --vider        # efface le graphe puis recharge

Entrées : data/extracted/json/<réf>.json, data/referentiel/*.csv
"""
import csv
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.evaluation.comparer import normaliser

load_dotenv()

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_JSON = RACINE / "data" / "extracted" / "json"
DOSSIER_REF = RACINE / "data" / "referentiel"

# Clé de chaque type de nœud (la propriété qui l'identifie)
CLES = {"Procedure": "reference", "Personne": "nom", "Acteur": "nom", "Direction": "nom",
        "Domaine": "nom", "Outil": "nom", "Projet": "code"}
# Types fermés : un nom absent du référentiel est marqué "à vérifier"
TYPES_FERMES = ("Direction", "Domaine", "Outil", "Projet")

# relation -> (type du sujet, type de l'objet)
TYPES_RELATIONS = {
    "A_POUR_PROPRIETAIRE": ("Procedure", "Personne"),
    "A_POUR_REDACTEUR": ("Procedure", "Personne"),
    "A_POUR_CONTRIBUTEUR": ("Procedure", "Personne"),
    "A_POUR_VALIDEUR": ("Procedure", "Personne"),
    "RATTACHEE_A_DIRECTION": ("Procedure", "Direction"),
    "APPARTIENT_AU_DOMAINE": ("Procedure", "Domaine"),
    "RATTACHEE_AU_PROJET": ("Procedure", "Projet"),
    "UTILISE_OUTIL": ("Procedure", "Outil"),
    "IMPLIQUE_ACTEUR": ("Procedure", "Acteur"),
    "PREVOIT_CONTROLE_PAR": ("Procedure", "Acteur"),
    "ALIMENTE": ("Procedure", "Procedure"),
    "DEPEND_DE": ("Procedure", "Procedure"),
    "COMPLETE": ("Procedure", "Procedure"),
    "OCCUPE_FONCTION": ("Personne", "Acteur"),
    "MEMBRE_DE": ("Personne", "Direction"),
}


def lire_csv(nom: str) -> list[dict]:
    with open(DOSSIER_REF / nom, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def charger_alias() -> dict:
    """Couche 1 : {(type, nom normalisé): nom officiel}, pour le nom officiel ET ses alias."""
    alias = {}
    for ligne in lire_csv("referentiel_alias.csv"):
        for variante in (ligne["nom_officiel"], ligne["alias"]):
            alias[(ligne["type"], normaliser(variante))] = ligne["nom_officiel"]
    return alias


def charger_libelles() -> dict:
    """Nom complet affiché pour les directions et les domaines : {(type, code): libellé}."""
    return {(l["type"], l["nom_officiel"]): l["alias"]
            for l in lire_csv("referentiel_alias.csv") if l["type"] in ("Direction", "Domaine")}


class Resolveur:
    """Donne pour chaque nom lu par le LLM le nom final du nœud, et s'il faut le vérifier."""

    def __init__(self, alias: dict, references_connues: set):
        self.alias = alias
        self.references = references_connues
        self.canon_acteur = {}       # clé normalisée d'un acteur -> écriture retenue (calculée sur le corpus)
        self.alias_ocr = {}          # même référentiel, comparé avec la clé tolérante à l'OCR
        for (t, cle), officiel in alias.items():
            k = (t, cle_ocr(cle))
            if self.alias_ocr.setdefault(k, officiel) != officiel:
                self.alias_ocr[k] = None          # deux noms officiels possibles : ambigu, on ne tranche pas

    def resoudre(self, type_: str, nom: str) -> tuple[str, bool]:
        nom = str(nom).strip()
        cle = normaliser(nom)
        if type_ == "Procedure":
            nom = nom.upper()
            return nom, nom not in self.references
        officiel = self.alias.get((type_, cle)) or self.alias_ocr.get((type_, cle_ocr(nom)))   # couches 1, 2 et 3
        if officiel:
            return officiel, False
        if type_ == "Acteur":                        # acteurs : pas de liste fermée, on regroupe d'après le corpus
            return self.canon_acteur.get(cle, nom), False
        return nom, type_ in TYPES_FERMES            # inconnu : jamais fusionné, marqué à vérifier


MODELE_REFERENCE = re.compile(r"^[A-Z]{2,3}-\d{2}$")     # ex. IT-05 : une référence de procédure
OCR = str.maketrans("l1", "ii")     # l, I et 1 sont souvent confondus par l'OCR


def cle_ocr(nom: str) -> str:
    """Clé de COMPARAISON seulement (jamais affichée) : "Idrissi" et "ldrissi" ont la même clé."""
    return normaliser(nom).translate(OCR)


class ResolveurPersonnes:
    """Décide, pour chaque nom de personne lu par le LLM, à quelle personne il correspond.

    Ordre de décision :
      1. nom identique à un nom de l'annuaire                      -> cette personne
      2. un seul candidat proche, contexte compatible, forme rare  -> cette personne (faute OCR probable)
      3. plusieurs candidats, ou contexte contradictoire           -> à valider (aucune fusion)
      4. aucun nom proche                                          -> à valider
    Sans annuaire : on écarte seulement codes et fonctions, et on regroupe les variantes
    d'après le corpus (la forme la plus fréquente gagne).
    """
    RARE = 2                         # une variante citée dans au plus 2 documents est "rare"

    def __init__(self, annuaire: list, alias: dict, noms_acteurs: set, mentions: dict, contexte: dict):
        self.annuaire = annuaire     # [{"nom", "cle", "ocr", "fonction", "direction"}]
        self.alias = alias
        self.noms_acteurs = noms_acteurs
        self.mentions = mentions     # clé normalisée -> {"graphie": ..., "docs": {références}}
        self.contexte = contexte     # clé normalisée -> {"fonction": set, "direction": set}

    def _compatible(self, cle: str, cand: dict) -> bool:
        ctx = self.contexte.get(cle, {})
        if ctx.get("fonction") and cand["fonction"] and cand["fonction"] not in ctx["fonction"]:
            return False
        if ctx.get("direction") and cand["direction"] and cand["direction"] not in ctx["direction"]:
            return False
        return True

    def resoudre(self, nom: str) -> tuple:
        """Retourne (nom final, None) ou (None, raison du refus)."""
        cle = normaliser(nom)
        for p in self.annuaire:                                    # règle 1 (prioritaire)
            if p["cle"] == cle:
                return p["nom"], None
        if MODELE_REFERENCE.match(nom.strip().upper()):
            return None, "code de procédure"
        if cle in self.noms_acteurs:
            return None, "fonction ou équipe (déjà dans la liste des acteurs)"
        if not self.annuaire:                                      # mode sans annuaire
            meme = [(len(m["docs"]), m["graphie"]) for k, m in self.mentions.items() if cle_ocr(k) == cle_ocr(nom)]
            return max(meme)[1], None
        candidats = []
        for p in self.annuaire:
            bruit = cle.replace(p["cle"], "").split()
            proche = (p["ocr"] == cle_ocr(nom)
                      or difflib.SequenceMatcher(None, cle, p["cle"]).ratio() >= 0.9
                      or (p["cle"] in cle and len(bruit) <= 1 and len("".join(bruit)) <= 3))
            if proche:
                candidats.append(p)
        if not candidats:
            return None, "absent de l'annuaire"                    # règle 4
        if len(candidats) > 1:
            return None, "plusieurs candidats : " + ", ".join(c["nom"] for c in candidats)   # règle 3
        if not self._compatible(cle, candidats[0]):
            return None, f"contexte contradictoire avec {candidats[0]['nom']}"               # règle 3
        if len(self.mentions[cle]["docs"]) > self.RARE:
            return None, f"écriture fréquente, proche de {candidats[0]['nom']}"             # règle 3
        return candidats[0]["nom"], None                           # règle 2


def charger_annuaire(alias: dict) -> list:
    """Personnel de la banque (équivalent d'un export RH). Fichier facultatif."""
    if not (DOSSIER_REF / "referentiel_personnes.csv").exists():
        return []
    annuaire = []
    for l in lire_csv("referentiel_personnes.csv"):
        direction = alias.get(("Direction", normaliser(l.get("direction", ""))), l.get("direction", ""))
        annuaire.append({"nom": l["nom"], "cle": normaliser(l["nom"]), "ocr": cle_ocr(l["nom"]),
                         "fonction": normaliser(l.get("fonction", "")), "direction": normaliser(direction)})
    return annuaire


def regrouper_acteurs(corpus: dict) -> dict:
    """Regroupe les écritures d'un même acteur d'après le corpus (aucune liste à maintenir).
    Deux écritures sont regroupées si elles ne diffèrent que par la casse, les accents ou la
    ponctuation, ou si l'une est l'abréviation de l'autre écrite entre parenthèses
    ("Responsable de la Sécurité ... (RSSI)" donne RSSI). L'écriture la plus citée est retenue."""
    docs = defaultdict(set)                  # écriture exacte -> procédures qui la citent
    for ref, d in corpus.items():
        for r in d["relations"]:
            if r["relation"] in TYPES_RELATIONS and TYPES_RELATIONS[r["relation"]][1] == "Acteur":
                docs[str(r["objet"]).strip()].add(ref)
    parent = {}

    def racine(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for ecriture in docs:
        racine(normaliser(ecriture))
        m = re.match(r"^(.*)\(([^()]+)\)\s*$", ecriture)
        if m and normaliser(m.group(1)):
            parent[racine(normaliser(m.group(2)))] = racine(normaliser(ecriture))   # l'abréviation rejoint le nom complet
            parent[racine(normaliser(m.group(1)))] = racine(normaliser(ecriture))   # le nom sans parenthèse aussi
    groupes = defaultdict(list)
    for ecriture in docs:
        groupes[racine(normaliser(ecriture))].append(ecriture)
    canon = {}
    for membres in groupes.values():
        choisie = max(membres, key=lambda e: (len(docs[e]), len(e)))
        for e in membres:
            canon[normaliser(e)] = choisie
    return canon


def preparer() -> dict:
    """Lit tous les JSON et retourne les nœuds et relations prêts à être écrits."""
    fichiers = sorted(p for p in DOSSIER_JSON.glob("*.json") if not p.stem.endswith("_brut"))
    references = {p.stem for p in fichiers}
    alias = charger_alias()
    res = Resolveur(alias, references)
    corpus = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in fichiers}
    res.canon_acteur = regrouper_acteurs(corpus)

    # Passe 1 : ce que le corpus dit des personnes (qui cite qui, avec quelle fonction)
    mentions, contexte, noms_acteurs, noms_personnes = {}, {}, set(), set()
    for ref, d in corpus.items():
        noms_acteurs |= {normaliser(a.get("nom", "")) for a in d.get("acteurs", [])}
        for r in d["relations"]:
            if r["relation"] not in TYPES_RELATIONS:
                continue
            for t, n in zip(TYPES_RELATIONS[r["relation"]], (r["sujet"], r["objet"])):
                if t == "Personne":
                    m = mentions.setdefault(normaliser(n), {"graphie": str(n).strip(), "docs": set()})
                    m["docs"].add(ref)
            if r["relation"] == "OCCUPE_FONCTION":
                noms_personnes.add(normaliser(r["sujet"]))     # le document donne son nom ET sa fonction
                contexte.setdefault(normaliser(r["sujet"]), {}).setdefault("fonction", set()).add(normaliser(r["objet"]))
            if r["relation"] == "MEMBRE_DE":
                direction = alias.get(("Direction", normaliser(r["objet"])), r["objet"])
                contexte.setdefault(normaliser(r["sujet"]), {}).setdefault("direction", set()).add(normaliser(direction))
    # une personne ne peut pas être aussi une fonction : on retire des acteurs les vrais noms de personnes
    # une fonction est dans la liste des acteurs et n'est jamais présentée comme une personne ;
    # les vrais noms de personnes figurent parfois aussi dans cette liste : on les en retire
    noms_fonctions = noms_acteurs - noms_personnes
    annuaire = charger_annuaire(alias)
    personnes = ResolveurPersonnes(annuaire, alias, noms_fonctions, mentions, contexte)

    # Passe 2 : construction des nœuds et des relations
    noeuds, relations, natures, ecartees = {}, [], {}, Counter()
    variantes = defaultdict(set)     # (type, nom final) -> écritures différentes rencontrées
    quarantaine = []                 # relations non chargées, avec la raison

    def noeud(type_, nom):
        nom_final, a_verifier = res.resoudre(type_, nom)
        cle = (type_, nom_final)
        if str(nom).strip() != nom_final:
            variantes[cle].add(str(nom).strip())
        noeuds[cle] = noeuds.get(cle, False) or a_verifier
        return nom_final

    for ref in sorted(references):
        noeud("Procedure", ref)
        d = corpus[ref]
        for a in d.get("acteurs", []):
            nom = res.resoudre("Acteur", a.get("nom", ""))[0]      # pas de nœud créé ici
            natures.setdefault(nom, str(a.get("nature", "")).strip())
        for r in d["relations"]:
            if r["relation"] not in TYPES_RELATIONS:
                ecartees[r["relation"]] += 1
                continue
            t_sujet, t_objet = TYPES_RELATIONS[r["relation"]]
            noms, refus = {}, None
            for cote, t, n in (("sujet", t_sujet, r["sujet"]), ("objet", t_objet, r["objet"])):
                if t == "Personne":
                    noms[cote], raison = personnes.resoudre(str(n))
                    if raison:
                        refus = (str(n).strip(), raison)
                else:
                    noms[cote] = None
            if refus:                # ni écrite dans le graphe, ni perdue : listée pour validation
                quarantaine.append({**r, "source": ref, "personne": refus[0], "raison": refus[1]})
                continue
            fin = {}
            for cote, t, n in (("sujet", t_sujet, r["sujet"]), ("objet", t_objet, r["objet"])):
                if t == "Personne":
                    fin[cote] = noms[cote]
                    if str(n).strip() != noms[cote]:
                        variantes[(t, noms[cote])].add(str(n).strip())
                    noeuds.setdefault((t, noms[cote]), False)
                else:
                    fin[cote] = noeud(t, n)
            relations.append({"t_sujet": t_sujet, "sujet": fin["sujet"], "relation": r["relation"],
                              "t_objet": t_objet, "objet": fin["objet"], "source": ref})
    return {"noeuds": noeuds, "relations": relations, "natures": natures, "ecartees": ecartees,
            "variantes": variantes, "libelles": charger_libelles(), "quarantaine": quarantaine,
            "taille_annuaire": len(annuaire)}


def bilan(p: dict) -> None:
    print("\nAnnuaire du personnel :", f"{p['taille_annuaire']} personnes" if p["taille_annuaire"]
          else "ABSENT (mode sans annuaire : data/referentiel/referentiel_personnes.csv introuvable)")
    print(f"\nNœuds : {len(p['noeuds'])}   Relations (avec doublons entre procédures) : {len(p['relations'])}")
    for t, n in sorted(Counter(t for t, _ in p["noeuds"]).items()):
        print(f"  {t:<10} {n}")
    a_verifier = sorted((t, n) for (t, n), v in p["noeuds"].items() if v)
    print(f"\nÀ vérifier ({len(a_verifier)}) : noms absents du référentiel")
    for t, n in a_verifier:
        print(f"  [{t}] {n}")
    print(f"\nRapprochements détectés ({len(p['variantes'])} entités avec des écritures différentes) :")
    for (t, nom), v in sorted(p["variantes"].items()):
        print(f"  [{t}] {nom}  <=  {', '.join(sorted(v))}")
    q = p["quarantaine"]
    groupes = Counter((x["personne"], x["raison"]) for x in q)
    print(f"\nRelations NON chargées : {len(q)} (personnes à valider : {len(groupes)})")
    for (nom, raison), n in sorted(groupes.items(), key=lambda g: (g[0][1], g[0][0])):
        print(f"  {nom}  x{n}   [{raison}]")
    if q:
        sortie = DOSSIER_JSON.parent / "a_verifier.csv"
        with open(sortie, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "sujet", "relation", "objet", "personne", "raison"], delimiter=";", extrasaction="ignore")
            w.writeheader()
            w.writerows(q)
        print(f"  (détail enregistré dans {sortie.relative_to(RACINE)})")
    if p["ecartees"]:
        print("Relations ignorées (type inconnu) :", dict(p["ecartees"]))


def charger(p: dict, vider: bool) -> None:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")))
    with driver, driver.session() as s:
        if vider:
            s.run("MATCH (n) DETACH DELETE n")
        for type_, cle in CLES.items():     # une clé = un seul nœud (c'est ce qui rend MERGE fiable)
            s.run(f"CREATE CONSTRAINT {type_.lower()}_cle IF NOT EXISTS "
                  f"FOR (n:{type_}) REQUIRE n.{cle} IS UNIQUE")
        for (type_, nom), a_verifier in p["noeuds"].items():
            s.run(f"MERGE (n:{type_} {{{CLES[type_]}: $nom}})", nom=nom)
            if a_verifier:
                s.run(f"MATCH (n:{type_} {{{CLES[type_]}: $nom}}) SET n.a_verifier = true", nom=nom)
        for (type_, nom), v in p["variantes"].items():       # écritures d'origine conservées
            s.run(f"MATCH (n:{type_} {{{CLES[type_]}: $nom}}) SET n.variantes = $v", nom=nom, v=sorted(v))
        for (type_, code), libelle in p["libelles"].items():  # nom complet pour l'affichage
            s.run(f"MATCH (n:{type_} {{nom: $code}}) SET n.libelle = $l", code=code, l=libelle)
        for nom, nature in p["natures"].items():
            s.run("MATCH (a:Acteur {nom: $nom}) SET a.nature = coalesce(a.nature, $nature)",
                  nom=nom, nature=nature)
        for r in p["relations"]:
            s.run(
                f"MATCH (a:{r['t_sujet']} {{{CLES[r['t_sujet']]}: $sujet}}), "
                f"(b:{r['t_objet']} {{{CLES[r['t_objet']]}: $objet}}) "
                f"MERGE (a)-[x:{r['relation']}]->(b) "
                "SET x.sources = CASE WHEN $source IN coalesce(x.sources, []) THEN x.sources "
                "ELSE coalesce(x.sources, []) + $source END",
                sujet=r["sujet"], objet=r["objet"], source=r["source"])
        # Données qui ne viennent pas des documents : référentiels
        for l in lire_csv("referentiel_projets.csv"):
            s.run("MERGE (p:Projet {code: $c}) SET p.description = $d", c=l["code"], d=l["description"])
        for l in lire_csv("referentiel_procedures.csv"):
            s.run("MATCH (p:Procedure {reference: $r}) SET p.titre = $t", r=l["reference"], t=l["titre"])


def main(args: list[str]) -> None:
    p = preparer()
    bilan(p)
    if "--simulation" in args:
        print("\nSimulation : rien n'a été écrit dans Neo4j.")
        return
    charger(p, vider="--vider" in args)
    print("\nChargement terminé. Ouvrez http://localhost:7474 pour vérifier.")


if __name__ == "__main__":
    main(sys.argv[1:])
