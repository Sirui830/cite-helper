"""PDF text extraction + cleanup + section splitting.

Port of LitReview+ A-side's parsing logic, minus chunking and clustering.
Inherits the same PDF noise characteristics (lost spaces, mid-word
linebreaks) — V0 does not aim to improve PDF cleanup, only to reuse a
known-acceptable parser.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


SECTION_HEADING_PATTERN = re.compile(
    r"""^
    \s*
    (?:\d+(?:\.\d+)*\.?\s*)?
    (
        introduction|background|abstract|
        methods?|methodology|materials\s+and\s+methods|data(?:\s+and\s+methods)?|
        procedure|participants|design|
        results?|findings?|analysis|
        discussion|implications|interpretation|
        conclusions?|concluding\s+remarks|summary|
        limitations?|future\s+(?:work|research|directions)|
        references|bibliography|acknowledgments?|acknowledgements?|
        funding|conflict\s+of\s+interest|author\s+contributions|
        data\s+availability\s+statement|ethics\s+statement|
        supplementary\s+material|
        literature\s+review|theoretical\s+framework|related\s+work
    )
    \s*[:.]?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


NOISE_SECTIONS = {
    "open access",
    "citation",
    "copyright",
    "keywords",
    "references",
    "acknowledgments",
    "acknowledgements",
    "funding",
    "conflict of interest",
    "author contributions",
    "data availability statement",
    "ethics statement",
    "supplementary material",
    "publisher's note",
    "bibliography",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 16)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return SECTION_HEADING_PATTERN.match(stripped) is not None


def clean_text(text: str) -> str:
    """Light cleanup that mirrors LitReview+ A-side."""
    text = text.replace(" ", " ")
    text = re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\*\*==> picture .*? omitted <==\*\*",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.[^\s]+", " ", text)
    text = re.sub(r"\*\*Vol\s+\d+[^*]*\*\*", " ", text)
    text = re.sub(r"\|[-: ]+\|", " ", text)
    return text.strip()


def normalize_section_key(section: str) -> str:
    s = re.sub(r"^\d+(\.\d+)*\.?\s*", "", section.strip())
    s = s.lower().strip(" :.")
    return s


def is_noise_section(section: str) -> bool:
    return normalize_section_key(section) in NOISE_SECTIONS


@dataclass
class PdfSection:
    paper_id: str
    pdf_path: str
    section: str
    text: str
    sha1: str
    is_noise: bool


def split_into_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Header"
    current_body: list[str] = []
    for raw_line in text.split("\n"):
        if looks_like_heading(raw_line):
            if current_body or sections:
                sections.append((current_title, current_body))
            current_title = raw_line.strip()
            current_body = []
        else:
            current_body.append(raw_line)
    if current_body or not sections:
        sections.append((current_title, current_body))
    return [(title, "\n".join(body).strip()) for title, body in sections]


def parse_pdf(pdf_path: Path) -> list[PdfSection]:
    """Parse one PDF into section-labeled text blocks."""
    try:
        import pymupdf4llm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm is required. Install: pip install pymupdf4llm"
        ) from exc

    raw = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    cleaned = clean_text(raw)
    paper_id = pdf_path.stem

    sections: list[PdfSection] = []
    for title, body in split_into_sections(cleaned):
        if not body.strip():
            continue
        sha1 = hashlib.sha1(body.encode("utf-8")).hexdigest()
        sections.append(
            PdfSection(
                paper_id=paper_id,
                pdf_path=str(pdf_path),
                section=title,
                text=body,
                sha1=sha1,
                is_noise=is_noise_section(title),
            )
        )
    return sections
