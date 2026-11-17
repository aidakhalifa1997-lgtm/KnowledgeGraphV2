"""Compare les relations extraites par le LLM avec la vérité terrain.

Usage (depuis la racine du dépôt, avec le .venv activé) :
    python -m src.evaluation.comparer IT-02    # détail pour une procédure
    python -m src.evaluation.comparer          # synthèse sur toutes les procédures extraites

Vérité terrain : data/reference/orizia_relations_reference.csv
  (les relations A_POUR_DESCRIPTION sont exclues : elles ne figurent pas dans les documents)
Extractions    : data/extracted/json/<reference>.json
"""
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
FICHIER_REFERENCE = RACINE / "data" / "reference" / "orizia_relations_reference.csv"
DOSSIER_JSON = RACINE / "data" / "extracted" / "json"
RELATION_EXCLUE = "A_POUR_DESCRIPTION"
# Relations sur les personnes : la référence les liste une seule fois (source "référentiel"),
# alors que le LLM les extrait dans chaque procédure. On les évalue donc globalement.
RELATIONS_PERSONNES = ("OCCUPE_FONCTION", "MEMBRE_DE")


def normaliser(texte: str) -> str:
    """Minuscules, sans accents, sans ponctuation superflue : pour comparer deux noms."""
    texte = unicodedata.normalize("NFD", str(texte).strip().lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^a-z0-9]+", " ", texte)
    return texte.strip()


def triplet(sujet, relation, objet) -> tuple:
    return (normaliser(sujet), relation.strip(), normaliser(objet))


def charger_reference() -> tuple[dict, dict]:
    """Retourne ({procédure: {triplet normalisé: triplet lisible}}, {mêmes triplets pour les personnes})."""
    par_doc, personnes = defaultdict(dict), {}
    with open(FICHIER_REFERENCE, encoding="utf-8-sig", newline="") as f:
        for ligne in csv.DictReader(f, delimiter=";"):
            if ligne["relation"] == RELATION_EXCLUE:
                continue
            t = triplet(ligne["sujet"], ligne["relation"], ligne["objet"])
            lisible = (ligne["sujet"], ligne["relation"], ligne["objet"])
            if ligne["relation"] in RELATIONS_PERSONNES:
                personnes[t] = lisible
            else:
                par_doc[ligne["source"]][t] = lisible
    return par_doc, personnes


def charger_extraction(reference: str) -> tuple[dict, dict]:
    """Retourne (relations de procédure, relations sur les personnes) extraites."""
    fichier = DOSSIER_JSON / f"{reference}.json"
    donnees = json.loads(fichier.read_text(encoding="utf-8"))
    procedure, personnes = {}, {}
    for r in donnees["relations"]:
        cible = personnes if r["relation"] in RELATIONS_PERSONNES else procedure
        cible[triplet(r["sujet"], r["relation"], r["objet"])] = (r["sujet"], r["relation"], r["objet"])
    return procedure, personnes


def comparer(attendu: dict, extrait: dict) -> tuple[set, set, set]:
    """Retourne (corrects, manquants, en trop)."""
    a, e = set(attendu), set(extrait)
    return a & e, a - e, e - a


def pourcentage(n: int, total: int) -> str:
    return f"{100 * n / total:.0f} %" if total else "-"


def main(cibles: list[str]) -> None:
    reference, ref_personnes = charger_reference()
    refs = cibles or sorted(p.stem for p in DOSSIER_JSON.glob("*.json") if not p.stem.endswith("_brut"))
    if not refs:
        print("Aucune extraction trouvée dans data/extracted/json.")
        return
    total_ok = total_att = total_ext = 0
    par_relation = defaultdict(lambda: [0, 0, 0])  # corrects, attendus, extraits
    extrait_personnes = {}
    for ref in refs:
        attendu = reference.get(ref, {})
        extrait, pers = charger_extraction(ref)
        extrait_personnes.update(pers)
        corrects, manquants, en_trop = comparer(attendu, extrait)
        total_ok += len(corrects); total_att += len(attendu); total_ext += len(extrait)
        for t in attendu: par_relation[t[1]][1] += 1
        for t in extrait: par_relation[t[1]][2] += 1
        for t in corrects: par_relation[t[1]][0] += 1
        print(f"{ref:8s} attendu={len(attendu):3d} extrait={len(extrait):3d} corrects={len(corrects):3d} "
              f"rappel={pourcentage(len(corrects), len(attendu))} précision={pourcentage(len(corrects), len(extrait))}")
        if cibles:  # détail seulement quand on demande une procédure précise
            print("\n-- MANQUANTS (dans la référence, pas extraits) --")
            for t in sorted(manquants): print("  ", attendu[t])
            print("\n-- EN TROP (extraits, pas dans la référence) --")
            for t in sorted(en_trop): print("  ", extrait[t])
    # Relations sur les personnes, évaluées globalement sur l'ensemble des procédures traitées
    pers_ok = set(extrait_personnes) & set(ref_personnes)
    print(f"\nPersonnes (OCCUPE_FONCTION, MEMBRE_DE) : extrait={len(extrait_personnes)} "
          f"corrects={len(pers_ok)} précision={pourcentage(len(pers_ok), len(extrait_personnes))} "
          f"(attendu sur le corpus entier : {len(ref_personnes)})")
    if cibles:
        for t in sorted(set(extrait_personnes) - set(ref_personnes)):
            print("   non reconnu :", extrait_personnes[t])
    print(f"\nTOTAL (relations de procédure) attendu={total_att} extrait={total_ext} corrects={total_ok} "
          f"rappel={pourcentage(total_ok, total_att)} précision={pourcentage(total_ok, total_ext)}")
    print("\nPar type de relation :")
    for rel, (ok, att, ext) in sorted(par_relation.items()):
        print(f"  {rel:24s} attendu={att:3d} extrait={ext:3d} corrects={ok:3d} "
              f"rappel={pourcentage(ok, att)} précision={pourcentage(ok, ext)}")


if __name__ == "__main__":
    main(sys.argv[1:])
