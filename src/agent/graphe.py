"""Nœud GraphRAG : retrouve dans le graphe Neo4j les faits utiles pour répondre à une question.

Étapes :
  1. comprendre()  : le LLM extrait l'intention (qui, quels outils...) et l'élément cité (« gestion des changements »).
  2. relier()      : l'élément cité est relié à un vrai nœud du graphe (comparaison avec les noms présents dans Neo4j :
                     aucune liste n'est écrite à la main).
  3. interroger()  : une requête Cypher (modèle) selon l'intention. Chaque ligne garde les procédures sources.
"""
import difflib
import json
import os
import re
import unicodedata
from functools import lru_cache

from neo4j import GraphDatabase

CLES = {"Procedure": "reference", "Personne": "nom", "Acteur": "nom", "Direction": "nom",
        "Domaine": "nom", "Outil": "nom", "Projet": "code"}
# relation qui rattache une procédure à chaque type de nœud
LIEN_PROCEDURE = {"Domaine": "APPARTIENT_AU_DOMAINE", "Outil": "UTILISE_OUTIL",
                  "Direction": "RATTACHEE_A_DIRECTION", "Projet": "RATTACHEE_AU_PROJET"}
ROLES = "A_POUR_PROPRIETAIRE|A_POUR_REDACTEUR|A_POUR_CONTRIBUTEUR|A_POUR_VALIDEUR"
ROLE_REL = {"proprietaire": "A_POUR_PROPRIETAIRE", "redacteur": "A_POUR_REDACTEUR",
            "valideur": "A_POUR_VALIDEUR", "contributeur": "A_POUR_CONTRIBUTEUR"}
LIBELLE_ROLE = {"A_POUR_PROPRIETAIRE": "propriétaire", "A_POUR_REDACTEUR": "rédacteur",
                "A_POUR_VALIDEUR": "valideur", "A_POUR_CONTRIBUTEUR": "contributeur"}
INTENTIONS = ("responsable", "intervenants", "outils", "dependances", "experts", "procedures_de", "similaires",
              "controles", "acteurs", "fiche")


# ------------------------------------------------------------------ accès à Neo4j
_driver = None


def requete(cypher: str, **params) -> list[dict]:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                                       auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")))
    with _driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]


# ------------------------------------------------------------------ 1. comprendre la question
CONSIGNE = """Tu analyses une question posée sur une base de procédures bancaires. Réponds par un JSON, rien d'autre :
{"intention": "...", "mention": "...", "type_mention": "...", "role": "..."}

intention (une seule) :
- responsable : qui est propriétaire, rédacteur, valideur ou contributeur d'une procédure (un rôle précis ou « responsable »)
- intervenants : qui s'occupe de, intervient dans, participe à une procédure (personnes et équipes impliquées)
- outils : quels outils utilise une procédure, OU quelles procédures utilisent un outil
- dependances : liens entre procédures (dépend de, alimente, complète)
- experts : qui sont les experts d'un domaine, d'un outil, d'une direction ou d'un projet
- procedures_de : lister les procédures d'un domaine, d'une direction, d'un outil, d'un projet ou d'une personne
- similaires : procédures proches d'une procédure donnée
- controles : contrôles prévus par une procédure
- acteurs : acteurs ou équipes impliqués dans une procédure
- fiche : informations générales sur une procédure

mention : l'élément cité dans la question, recopié tel quel (nom ou référence), sans article.
type_mention : procedure, outil, domaine, direction, personne ou projet.
role : seulement pour « responsable » : proprietaire, redacteur, valideur ou contributeur si la question vise un rôle précis ; sinon vide.

Exemples :
Qui a rédigé la procédure de gestion des accès ? -> {"intention": "responsable", "mention": "gestion des accès", "type_mention": "procedure", "role": "redacteur"}
Qui valide la procédure paie ? -> {"intention": "responsable", "mention": "paie", "type_mention": "procedure", "role": "valideur"}
Qui s'occupe de la gestion des habilitations ? -> {"intention": "intervenants", "mention": "gestion des habilitations", "type_mention": "procedure", "role": ""}
Quels outils sont utilisés par FIN-03 ? -> {"intention": "outils", "mention": "FIN-03", "type_mention": "procedure", "role": ""}
Qui maîtrise le domaine conformité ? -> {"intention": "experts", "mention": "conformité", "type_mention": "domaine", "role": ""}
Que fait la direction juridique ? -> {"intention": "procedures_de", "mention": "juridique", "type_mention": "direction", "role": ""}"""


def comprendre(question: str, appeler_llm) -> dict:
    """Retourne {"intention", "mention", "type_mention"}. Valeurs vides si la réponse est illisible."""
    brut = appeler_llm(CONSIGNE, question, max_tokens=120)
    m = re.search(r"\{.*?\}", brut, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    intention = d.get("intention") if d.get("intention") in INTENTIONS else ""
    role = str(d.get("role") or "").strip().lower()
    return {"intention": intention, "mention": str(d.get("mention") or "").strip(),
            "type_mention": str(d.get("type_mention") or "").strip().capitalize(),
            "role": role if role in ROLE_REL else ""}


# ------------------------------------------------------------------ 2. relier la mention à un nœud du graphe
def normaliser(texte: str) -> str:
    t = unicodedata.normalize("NFD", str(texte).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


@lru_cache(maxsize=1)
def catalogue() -> list[dict]:
    """Tous les noms présents dans le graphe (lu dans Neo4j, jamais écrit à la main)."""
    lignes = requete("""
        MATCH (n) WHERE n:Procedure OR n:Personne OR n:Direction OR n:Domaine OR n:Outil OR n:Projet
        RETURN labels(n)[0] AS type,
               CASE labels(n)[0] WHEN 'Procedure' THEN n.reference WHEN 'Projet' THEN n.code ELSE n.nom END AS cle,
               [x IN [n.titre, n.libelle, n.description] WHERE x IS NOT NULL] + coalesce(n.variantes, []) AS textes""")
    return [{"type": l["type"], "cle": l["cle"], "textes": [normaliser(t) for t in [l["cle"]] + l["textes"] if t]}
            for l in lignes if l["cle"]]


def score(mention: str, texte: str) -> float:
    """1.0 identique ; 0.9 les mots de l'un sont tous dans l'autre ; sinon ressemblance (tolère les fautes de frappe)."""
    if mention == texte:
        return 1.0
    a, b = set(mention.split()), set(texte.split())
    if a and b and (a <= b or b <= a) and min(len(mention), len(texte)) >= 3:
        return 0.9
    return difflib.SequenceMatcher(None, mention, texte).ratio()


def relier(mention: str, type_mention: str, noeuds: list[dict] | None = None, seuil: float = 0.85) -> list[dict]:
    """Retourne les nœuds correspondants (plusieurs si ex æquo : la question est alors ambiguë)."""
    m = normaliser(mention)
    if not m:
        return []
    noeuds = noeuds if noeuds is not None else catalogue()
    if type_mention in CLES:                                  # on se limite au type annoncé s'il existe
        noeuds = [n for n in noeuds if n["type"] == type_mention] or noeuds
    notes = [(max(score(m, t) for t in n["textes"]), n) for n in noeuds if n["textes"]]
    meilleur = max((s for s, _ in notes), default=0)
    if meilleur < seuil:
        return []
    return [{"type": n["type"], "cle": n["cle"], "score": round(s, 2)} for s, n in notes if s == meilleur][:3]


def relier_par_llm(mention: str, type_mention: str, appeler_llm, noeuds: list[dict] | None = None) -> list[dict]:
    """Repli quand la comparaison de texte échoue (abréviation, reformulation : « sécurité des SI »).
    Le LLM choisit parmi les éléments RÉELS du graphe ; il ne peut pas en inventer."""
    noeuds = noeuds if noeuds is not None else catalogue()
    candidats = [n for n in noeuds if n["type"] == type_mention]
    if not candidats or len(candidats) > 60:
        return []
    liste = "\n".join(f"{i}. {n['cle']} : {n['textes'][-1] if len(n['textes']) > 1 else ''}" for i, n in enumerate(candidats, 1))
    consigne = ("Voici des éléments d'un graphe de connaissances. Indique le numéro de celui qui correspond à "
                "l'expression donnée, ou 0 si aucun ne correspond vraiment. Réponds par le numéro seul.\n" + liste)
    m = re.search(r"\d+", appeler_llm(consigne, mention, max_tokens=10))
    i = int(m.group(0)) if m else 0
    return [{"type": candidats[i - 1]["type"], "cle": candidats[i - 1]["cle"], "score": 0.5}] if 1 <= i <= len(candidats) else []


# ------------------------------------------------------------------ 3. interroger le graphe
def regrouper(lignes: list[dict]) -> list[dict]:
    """Une seule ligne par personne, avec tous ses rôles (évite « Nadia Berrada (rédacteur) et Nadia Berrada (propriétaire) »)."""
    par_personne = {}
    for l in lignes:
        p = par_personne.setdefault(l["personne"], {"personne": l["personne"], "roles": [], "sources": l["sources"]})
        p["roles"].append(l["role"])
    return list(par_personne.values())


def personnes(cle: str, intention: str, role: str) -> tuple[list[dict], str]:
    """Personnes liées à une procédure. Retourne (lignes, remarque).
    - rôle précis demandé (valideur...) : on ne renvoie que ce rôle ; s'il n'existe pas, on le dit.
    - « responsable » sans rôle : le propriétaire ; « intervenants » : tous les rôles + les équipes impliquées."""
    tous = requete(f"""MATCH (p:Procedure {{reference: $cle}})-[r:{ROLES}]->(x:Personne)
                       RETURN type(r) AS code, x.nom AS personne, [p.reference] AS sources""", cle=cle)
    for l in tous:
        l["role"] = LIBELLE_ROLE[l.pop("code")]
    if intention == "intervenants":
        equipes = requete("""MATCH (p:Procedure {reference: $cle})-[:IMPLIQUE_ACTEUR]->(a:Acteur)
                             RETURN a.nom AS equipe, [p.reference] AS sources""", cle=cle)
        return regrouper(tous) + equipes, ""
    voulu = LIBELLE_ROLE[ROLE_REL[role]] if role else "propriétaire"
    ciblees = [l for l in tous if l["role"] == voulu]
    if ciblees:
        return regrouper(ciblees), ""
    return regrouper(tous), f"Aucun {voulu} n'est enregistré pour {cle} dans le graphe. Voici les autres personnes liées à cette procédure."


def interroger(intention: str, noeud: dict) -> list[dict]:
    """Une requête par intention. Chaque ligne contient les références de procédures sources."""
    type_, cle = noeud["type"], noeud["cle"]
    est_proc = type_ == "Procedure"

    if intention == "outils":
        if est_proc:
            return requete("""MATCH (p:Procedure {reference: $cle})-[r:UTILISE_OUTIL]->(o:Outil)
                              RETURN o.nom AS outil, [p.reference] AS sources""", cle=cle)
        if type_ == "Outil":
            return requete("""MATCH (p:Procedure)-[:UTILISE_OUTIL]->(o:Outil {nom: $cle})
                              RETURN p.reference AS procedure, p.titre AS titre, [p.reference] AS sources
                              ORDER BY procedure""", cle=cle)

    if intention == "dependances" and est_proc:
        lignes = requete("""MATCH (p:Procedure {reference: $cle})-[r:DEPEND_DE|ALIMENTE|COMPLETE]-(q:Procedure)
                            RETURN type(r) AS relation, startNode(r) = p AS sortant,
                                   q.reference AS procedure, q.titre AS titre,
                                   [p.reference, q.reference] AS sources""", cle=cle)
        verbes = {"DEPEND_DE": "dépend de", "ALIMENTE": "alimente", "COMPLETE": "complète"}
        for l in lignes:                                  # le sens est écrit en toutes lettres
            a, b = (cle, l["procedure"]) if l.pop("sortant") else (l["procedure"], cle)
            l["lien"] = f"{a} {verbes[l.pop('relation')]} {b}"
        return lignes

    if intention in ("experts", "responsable", "intervenants") and type_ in LIEN_PROCEDURE:
        cle_prop = CLES[type_]
        return requete(f"""MATCH (n:{type_} {{{cle_prop}: $cle}})<-[:{LIEN_PROCEDURE[type_]}]-(p:Procedure)
                                 -[r:{ROLES}]->(x:Personne)
                           RETURN x.nom AS personne, collect(DISTINCT p.reference) AS sources,
                                  collect(DISTINCT type(r)) AS roles
                           ORDER BY size(sources) DESC LIMIT 10""", cle=cle)

    if intention == "procedures_de":
        if type_ in LIEN_PROCEDURE:
            return requete(f"""MATCH (n:{type_} {{{CLES[type_]}: $cle}})<-[:{LIEN_PROCEDURE[type_]}]-(p:Procedure)
                               RETURN p.reference AS procedure, p.titre AS titre, [p.reference] AS sources
                               ORDER BY procedure""", cle=cle)
        if type_ == "Personne":
            return requete(f"""MATCH (p:Procedure)-[r:{ROLES}]->(x:Personne {{nom: $cle}})
                               RETURN p.reference AS procedure, p.titre AS titre, type(r) AS role,
                                      [p.reference] AS sources ORDER BY procedure""", cle=cle)

    if intention == "similaires" and est_proc:
        return requete("""MATCH (p:Procedure {reference: $cle})-[r1:APPARTIENT_AU_DOMAINE|UTILISE_OUTIL|RATTACHEE_A_DIRECTION|RATTACHEE_AU_PROJET]->(x)
                                <-[r2]-(q:Procedure)
                          WHERE q <> p AND type(r1) = type(r2)
                          RETURN q.reference AS procedure, q.titre AS titre,
                                 collect(DISTINCT coalesce(x.nom, x.code)) AS en_commun,
                                 [p.reference, q.reference] AS sources
                          ORDER BY size(en_commun) DESC LIMIT 8""", cle=cle)

    if intention == "controles" and est_proc:
        return requete("""MATCH (p:Procedure {reference: $cle})-[:PREVOIT_CONTROLE_PAR]->(a:Acteur)
                          RETURN a.nom AS acteur, [p.reference] AS sources""", cle=cle)

    if intention == "acteurs" and est_proc:
        return requete("""MATCH (p:Procedure {reference: $cle})-[:IMPLIQUE_ACTEUR]->(a:Acteur)
                          RETURN a.nom AS acteur, [p.reference] AS sources""", cle=cle)

    if est_proc:                                              # « fiche » ou intention non gérée : tout ce qui est relié
        return requete("""MATCH (p:Procedure {reference: $cle})-[r]-(x)
                          RETURN type(r) AS relation, startNode(r) = p AS sortant, labels(x)[0] AS type,
                                 CASE labels(x)[0] WHEN 'Procedure' THEN x.reference WHEN 'Projet' THEN x.code
                                      ELSE x.nom END AS valeur, [p.reference] AS sources""", cle=cle)
    return []


# ------------------------------------------------------------------ point d'entrée du nœud
def chercher(question: str, appeler_llm) -> dict:
    """Retourne {"intention", "noeuds", "lignes", "sources", "diagnostic"} ; lignes vides si rien trouvé."""
    c = comprendre(question, appeler_llm)
    resultat = {"intention": c["intention"], "mention": c["mention"], "noeuds": [], "lignes": [],
                "sources": [], "diagnostic": "", "remarque": ""}
    if not c["intention"] or not c["mention"]:
        resultat["diagnostic"] = "question non comprise par l'analyse"
        return resultat
    noeuds = relier(c["mention"], c["type_mention"]) or relier_par_llm(c["mention"], c["type_mention"], appeler_llm)
    resultat["noeuds"] = noeuds
    if not noeuds:
        resultat["diagnostic"] = f"« {c['mention']} » ne correspond à aucun élément du graphe"
        return resultat
    for noeud in noeuds:
        if c["intention"] in ("responsable", "intervenants") and noeud["type"] == "Procedure":
            lignes, remarque = personnes(noeud["cle"], c["intention"], c["role"])
            resultat["remarque"] += remarque + " "
        else:
            lignes = interroger(c["intention"], noeud)
        for ligne in lignes:
            ligne["element"] = f"{noeud['type']} {noeud['cle']}"      # pour distinguer les cas ambigus
            resultat["lignes"].append(ligne)
    resultat["sources"] = sorted({s for l in resultat["lignes"] for s in l.get("sources", [])})
    if not resultat["lignes"]:
        resultat["diagnostic"] = "le graphe ne contient aucun résultat pour cette question"
    return resultat
