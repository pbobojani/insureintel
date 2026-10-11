"""
Text extractor for uploaded regulatory circular documents.
Supports PDF, DOCX, and plain text formats.
Requires: pdfplumber (PDF), python-docx (DOCX)
"""
from __future__ import annotations

import re
from pathlib import Path


def extract_text(file_path: str | Path) -> str:
    """
    Extract raw text from a document file.
    Returns empty string if extraction fails or format unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        return ""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in (".docx", ".doc"):
        return _extract_docx(path)
    if suffix in (".txt", ".text"):
        return _extract_txt(path)

    return _extract_txt(path)


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return ""
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_texts.append(row_text)
        return "\n".join(paragraphs + table_texts)
    except ImportError:
        return ""
    except Exception:
        return ""


def _extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, Exception):
            continue
    return ""


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def extract_and_translate(file_path: str | Path, document_id: int, db) -> str:
    """Backward-compatible wrapper for callers that still use the old name."""
    return extract_text(file_path)
