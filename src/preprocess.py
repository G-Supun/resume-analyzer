from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving useful structure.
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # remove repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # normalize line spacing
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove zero-width and unusual invisible chars
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # standardize long dashes
    text = text.replace("—", "-").replace("–", "-")

    return text.strip()


def normalize_spaces(text: str) -> str:
    """
    Lighter normalization helper used after cleaning.
    """
    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_nonempty_lines(text: str) -> list[str]:
    """
    Split text into cleaned non-empty lines.
    """
    if not text:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def build_model_ready_text(text: str, max_chars: int = 2500) -> str:
    """
    Create a shorter version of the text for semantic matching.
    """
    cleaned = clean_text(text)
    cleaned = normalize_spaces(cleaned)

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].strip()