from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def extract_text_from_pdf_or_doc(path: PathLike) -> str:
    """
    Extract text from PDF, TXT, or DOCX files.

    Supported:
    - .pdf  -> PyMuPDF primary, pdfplumber fallback
    - .txt  -> plain text read
    - .docx -> python-docx

    Returns cleaned raw text string.
    Raises FileNotFoundError or ValueError on failure.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return text.strip()

    if suffix == ".docx":
        return _extract_from_docx(file_path)

    if suffix == ".pdf":
        text = _extract_from_pdf_pymupdf(file_path)
        if len(text.strip()) < 50:
            text = _extract_from_pdf_pdfplumber(file_path)
        if not text.strip():
            raise ValueError(f"Could not extract text from PDF: {file_path}")
        return text.strip()

    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_from_docx(file_path: Path) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx is not installed. Run: python -m pip install python-docx"
        ) from e

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text from {file_path}: {e}") from e


def _extract_from_pdf_pymupdf(file_path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is not installed. Run: python -m pip install pymupdf"
        ) from e

    try:
        pages = []
        with fitz.open(file_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        return "\n".join(pages).strip()
    except Exception:
        return ""


def _extract_from_pdf_pdfplumber(file_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber is not installed. Run: python -m pip install pdfplumber"
        ) from e

    try:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""