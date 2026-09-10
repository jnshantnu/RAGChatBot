"""PDF text extraction. Page-level granularity: each PDF page becomes one
chunk (further split if unusually long), since these reference manuals don't
carry markdown-style headings to chunk on structurally.
"""
from pypdf import PdfReader

MAX_CHUNK_CHARS = 3000


def extract_pages(path: str) -> list[str]:
    # One string per page, in order. `extract_text()` can return None for a
    # page with no extractable text (e.g. a scanned image) -- normalize to "".
    reader = PdfReader(path)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def split_long_page(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Splits an overlong page on paragraph boundaries into <= max_chars pieces."""
    if len(text) <= max_chars:
        return [text]

    # Greedily pack whole paragraphs into a piece until adding the next one
    # would exceed max_chars, then start a new piece -- never splits a
    # paragraph mid-sentence.
    pieces = []
    current = []
    current_len = 0
    for para in text.split("\n\n"):
        para_len = len(para) + 2  # +2 accounts for the "\n\n" that joins pieces back together
        if current_len + para_len > max_chars and current:
            pieces.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += para_len
    if current:
        pieces.append("\n\n".join(current))
    return pieces
