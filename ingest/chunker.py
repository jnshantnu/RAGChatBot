import hashlib
import re
from dataclasses import dataclass, field

from ingest.pdf_loader import split_long_page


# In-memory representation of one chunk, before it's embedded and written to
# the `chunks` table. Both chunk_markdown and chunk_pdf build lists of these.
@dataclass
class Chunk:
    chunk_id: str      # stable hash-based id, see stable_chunk_id()
    doc_id: str         # parent document's id, e.g. "buysell-pws-get-invoice-service-reference-manual"
    ordinal: int        # this chunk's position within its document (0-indexed)
    heading: str        # human-readable label shown in citations/UI, e.g. "Page 6" or a markdown heading
    text: str            # the actual text that gets embedded and stored
    metadata: dict = field(default_factory=dict)   # category/page/guaranteed flags -- see chunk_pdf/chunk_markdown
    acl: list = field(default_factory=lambda: ["public"])   # groups allowed to retrieve this chunk


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Tiny frontmatter parser: `key: value` and `key: [a, b]` lines between --- fences."""
    # Match the --- ... --- block at the top of the file; if there isn't one,
    # treat the whole input as body with no frontmatter fields.
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
        # "acl: [public, role:principal]" -> a Python list; anything else stays a plain string.
        if value.startswith("[") and value.endswith("]"):
            fm[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            fm[key] = value
    return fm, body


def stable_chunk_id(doc_id: str, heading: str, ordinal: int) -> str:
    # Deterministic id: re-ingesting the same doc_id/heading/ordinal always
    # produces the same chunk_id, so re-running ingestion on unchanged content
    # doesn't create duplicate rows or churn foreign keys elsewhere.
    digest = hashlib.sha256(f"{doc_id}::{heading}::{ordinal}".encode()).hexdigest()
    return f"{doc_id}-{digest[:12]}"


def chunk_markdown(raw_text: str, fallback_doc_id: str) -> list[Chunk]:
    """Structure-aware chunking: split on markdown headings (## and #), one chunk per section."""
    frontmatter, body = _parse_frontmatter(raw_text)
    doc_id = frontmatter.get("doc_id", fallback_doc_id)
    acl = frontmatter.get("acl", ["public"])
    category = frontmatter.get("category", "")
    # guaranteed: true -- chunk is embedded/searched normally, but is also
    # fetched unconditionally on a separate path (see fetch_guaranteed_chunks
    # in hybrid_search.py) so it can never lose the top-20 rerank competition
    # and silently vanish from an answer it should have informed.
    guaranteed = frontmatter.get("guaranteed", "false").strip().lower() == "true"

    # Split on lines starting with '#' (any heading level), keeping the heading with its section.
    sections = re.split(r"\n(?=#{1,6}\s)", body.strip())

    chunks = []
    for ordinal, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Pull the heading text out of the section's first line (e.g. "## Downgrade" -> "Downgrade")
        # for use as this chunk's human-readable label; fall back to the doc id if there's no heading.
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
                metadata={"category": category, "guaranteed": guaranteed},
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
        # Normally one piece (the whole page); split_long_page only returns
        # more than one when the page's text exceeds MAX_CHUNK_CHARS.
        pieces = split_long_page(page_text)
        for piece_idx, piece in enumerate(pieces):
            # Build a readable heading: page number, plus the page's first
            # non-blank line as a rough label, plus a part marker if this page
            # got split into multiple pieces.
            first_line = next((line.strip() for line in piece.splitlines() if line.strip()), "")
            heading = f"Page {page_num}"
            if first_line:
                heading += f" — {first_line[:70]}"
            if len(pieces) > 1:
                heading += f" (part {piece_idx + 1}/{len(pieces)})"
            chunk_id = stable_chunk_id(doc_id, f"p{page_num}-{piece_idx}", ordinal)
            # The doc-title prefix (see the docstring above) is prepended to the
            # stored/embedded text here, after splitting -- so it doesn't count
            # against MAX_CHUNK_CHARS and doesn't affect where a page gets split.
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
