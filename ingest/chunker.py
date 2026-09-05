import hashlib
import re
from dataclasses import dataclass, field

from ingest.pdf_loader import split_long_page


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    heading: str
    text: str
    metadata: dict = field(default_factory=dict)
    acl: list = field(default_factory=lambda: ["public"])


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Tiny frontmatter parser: `key: value` and `key: [a, b]` lines between --- fences."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        return {}, raw
    fm_text, body = match.groups()
    fm: dict = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            fm[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            fm[key] = value
    return fm, body


def stable_chunk_id(doc_id: str, heading: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{doc_id}::{heading}::{ordinal}".encode()).hexdigest()
    return f"{doc_id}-{digest[:12]}"


def chunk_markdown(raw_text: str, fallback_doc_id: str) -> list[Chunk]:
    """Structure-aware chunking: split on markdown headings (## and #), one chunk per section."""
    frontmatter, body = _parse_frontmatter(raw_text)
    doc_id = frontmatter.get("doc_id", fallback_doc_id)
    acl = frontmatter.get("acl", ["public"])
    category = frontmatter.get("category", "")

    # Split on lines starting with '#' (any heading level), keeping the heading with its section.
    sections = re.split(r"\n(?=#{1,6}\s)", body.strip())

    chunks = []
    for ordinal, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^#{1,6}\s+(.*)", section)
        heading = heading_match.group(1).strip() if heading_match else fallback_doc_id
        chunk_id = stable_chunk_id(doc_id, heading, ordinal)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                ordinal=ordinal,
                heading=heading,
                text=section,
                metadata={"category": category},
                acl=acl,
            )
        )
    return chunks


def chunk_pdf(pages: list[str], doc_id: str, acl: list[str], category: str = "", doc_title: str = "") -> list[Chunk]:
    """One chunk per page (further split if unusually long). Keyed on page number
    and piece index, not heading text, since extracted text can vary slightly
    between runs (whitespace, ligatures) even when the underlying page hasn't.

    Each chunk's stored text is prefixed with `doc_title` -- a page in isolation
    often can't say which document (or business model) it belongs to (e.g. a
    filename's "BuySell" vs "NxM" distinction may never appear in the page text
    itself), so without this prefix neither the embedding nor the keyword arm
    nor the reranker has any way to know. Standard "contextual retrieval" fix
    for context lost at the chunk boundary.
    """
    chunks = []
    ordinal = 0
    for page_num, page_text in enumerate(pages, start=1):
        if len(page_text) < 20:  # blank or image-only page -- nothing to retrieve
            continue
        pieces = split_long_page(page_text)
        for piece_idx, piece in enumerate(pieces):
            first_line = next((line.strip() for line in piece.splitlines() if line.strip()), "")
            heading = f"Page {page_num}"
            if first_line:
                heading += f" — {first_line[:70]}"
            if len(pieces) > 1:
                heading += f" (part {piece_idx + 1}/{len(pieces)})"
            chunk_id = stable_chunk_id(doc_id, f"p{page_num}-{piece_idx}", ordinal)
            text = f"[{doc_title}]\n\n{piece}" if doc_title else piece
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    ordinal=ordinal,
                    heading=heading,
                    text=text,
                    metadata={"category": category, "page": page_num},
                    acl=acl,
                )
            )
            ordinal += 1
    return chunks
