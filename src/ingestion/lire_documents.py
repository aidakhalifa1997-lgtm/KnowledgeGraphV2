"""Lecture des documents du corpus (Word, PDF natif, PDF scanné) et conversion en texte.

Usage (depuis la racine du dépôt) :
    python -m src.ingestion.lire_documents                 # lit tout data/raw
    python -m src.ingestion.lire_documents data/raw/word   # lit un dossier ou un fichier

Le texte obtenu est écrit dans data/extracted/<nom>.txt
"""
import io
import sys
from pathlib import Path

import docx                      # python-docx : lecture des fichiers Word
import pymupdf                   # PyMuPDF : lecture des PDF et rendu des pages en images
import pytesseract               # OCR (nécessite le programme Tesseract installé)
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_SORTIE = RACINE / "data" / "extracted"
SEUIL_TEXTE_PAR_PAGE = 50        # en dessous, on considère la page comme une image (scan)
LANGUE_OCR = "fra"


def lire_word(chemin: Path) -> str:
    """Lit un .docx : paragraphes et tableaux, dans l'ordre du document."""
    document = docx.Document(chemin)
    lignes = []
    for element in document.element.body.iterchildren():
        if element.tag.endswith("}p"):
            texte = Paragraph(element, document).text.strip()
            if texte:
                lignes.append(texte)
        elif element.tag.endswith("}tbl"):
            for ligne in Table(element, document).rows:
                cellules = []
                for cellule in ligne.cells:
                    if cellule.text.strip() not in cellules:   # évite les doublons des cellules fusionnées
                        cellules.append(cellule.text.strip())
                lignes.append(" | ".join(cellules))
    return "\n".join(lignes)


def ocr_page(page) -> str:
    """Transforme une page PDF en image puis la lit avec Tesseract."""
    image_png = page.get_pixmap(dpi=200).tobytes("png")
    image = Image.open(io.BytesIO(image_png))
    return pytesseract.image_to_string(image, lang=LANGUE_OCR)


def lire_pdf(chemin: Path) -> tuple[str, str]:
    """Lit un PDF. Retourne (texte, mode) avec mode = 'natif', 'ocr' ou 'mixte'."""
    pages, modes = [], set()
    with pymupdf.open(chemin) as pdf:
        for page in pdf:
            texte = page.get_text().strip()
            if len(texte) >= SEUIL_TEXTE_PAR_PAGE:
                modes.add("natif")
            else:
                texte = ocr_page(page).strip()
                modes.add("ocr")
            pages.append(texte)
    mode = modes.pop() if len(modes) == 1 else "mixte"
    return "\n\n".join(pages), mode


def lire_document(chemin: Path) -> tuple[str, str]:
    """Retourne (texte, mode de lecture) selon l'extension du fichier."""
    extension = chemin.suffix.lower()
    if extension == ".docx":
        return lire_word(chemin), "word"
    if extension == ".pdf":
        return lire_pdf(chemin)
    raise ValueError(f"Format non géré : {chemin.name}")


def main(cibles: list[str]) -> None:
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    fichiers = []
    for cible in cibles or [str(RACINE / "data" / "raw")]:
        p = Path(cible)
        fichiers += sorted(p.rglob("*")) if p.is_dir() else [p]
    fichiers = [f for f in fichiers if f.suffix.lower() in (".docx", ".pdf")]
    if not fichiers:
        print("Aucun fichier .docx ou .pdf trouvé.")
        return
    for f in fichiers:
        try:
            texte, mode = lire_document(f)
        except Exception as erreur:
            print(f"ERREUR {f.name} : {erreur}")
            continue
        (DOSSIER_SORTIE / f"{f.stem}.txt").write_text(texte, encoding="utf-8")
        print(f"{f.name:60s} mode={mode:6s} {len(texte):6d} caractères")


if __name__ == "__main__":
    main(sys.argv[1:])
