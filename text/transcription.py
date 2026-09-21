"""Annotation loading and proxy-to-Greek transcription helpers."""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

from pipeline_config import DatasetLayout


PROXY_TO_GREEK = {
    "A": "Α", "B": "Β", "G": "Γ", "D": "Δ", "E": "Ε", "Z": "Ζ",
    "H": "Η", "Q": "Θ", "I": "Ι", "K": "Κ", "L": "Λ", "M": "Μ",
    "N": "Ν", "X": "Ξ", "O": "Ο", "P": "Π", "R": "Ρ", "S": "Σ",
    "T": "Τ", "Y": "Υ", "F": "Φ", "C": "Χ", "U": "Ψ", "W": "Ω",
    ":": ":", ")": ")", "&": "&", "*": "*",
}


def read_annotation_text(path: Path) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"could not read {path}")


def extract_proxy_transcript(file_text: str) -> str:
    lines = file_text.splitlines()
    start_idx = None

    for index, line in enumerate(lines):
        if line.strip() == "# Transcript":
            start_idx = index + 1
            break

    if start_idx is None:
        warnings.warn("Could not find '# Transcript'; using full file text.", stacklevel=2)
        return file_text

    end_idx = len(lines)
    for index in range(start_idx, len(lines)):
        if lines[index].lstrip().startswith("#"):
            end_idx = index
            break

    transcript_lines = lines[start_idx:end_idx]
    while transcript_lines and transcript_lines[-1].strip() == "":
        transcript_lines.pop()
    return "\n".join(transcript_lines)


def convert_proxy_to_greek(text: str) -> str:
    return "".join(char if char.isspace() else PROXY_TO_GREEK[char] for char in text)


def normalize_greek(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.upper()


def load_annotations(layout: DatasetLayout, valid_pairs: set[str]):
    import pandas as pd

    rows = []
    for inscription_id in sorted(valid_pairs):
        path = layout.annotations_dir / f"{inscription_id}_letters.txt"
        if not path.exists():
            continue

        file_text, encoding = read_annotation_text(path)
        proxy_text = extract_proxy_transcript(file_text)
        rows.append(
            {
                "filename": path.name,
                "inscription_id": inscription_id,
                "proxy_text": proxy_text,
                "greek_text": convert_proxy_to_greek(proxy_text),
                "encoding": encoding,
            }
        )

    return pd.DataFrame(rows)

