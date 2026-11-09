from pathlib import Path

from src.ingestion.lire_documents import lire_word

RACINE = Path(__file__).resolve().parents[1]


def test_lecture_word_contient_le_titre():
    fichiers = sorted((RACINE / "data" / "raw" / "word").glob("*.docx"))
    if not fichiers:
        return  # pas de corpus copié : le test est ignoré
    texte = lire_word(fichiers[0])
    assert "Procédure" in texte
    assert "Règles de gestion" in texte
