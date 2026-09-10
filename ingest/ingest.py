"""CLI: walk the corpus dir -> chunk -> embed -> upsert into Postgres.

Supports markdown (frontmatter-driven ACL/category) and PDF (page-level
chunking, ACL/category assigned by filename convention -- see
_pdf_acl_and_category below). Idempotent: a document's content_hash (raw file
bytes) is compared against the `documents` registry; unchanged files are
skipped entirely, changed ones have their old chunks deleted (one DELETE
clears both the GIN and HNSW index entries atomically) and are
re-chunked/re-embedded from scratch.
"""
import hashlib
import os
import re
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ingest.chunker import Chunk, chunk_markdown, chunk_pdf
from ingest.embed import embed_texts
from ingest.pdf_loader import extract_pages

EMBED_BATCH_SIZE = 32


def _vector_literal(values: list[float]) -> str:
    # psycopg has no native Python type for pgvector's `vector` column, so we
    # format the embedding as the literal text pgvector expects ("[0.1,0.2,...]")
    # and let the SQL cast it with `::vector` at insert time.
    return "[" + ",".join(repr(v) for v in values) + "]"


def _content_hash_bytes(raw: bytes) -> str:
    # Hashing raw bytes (not extracted text) means a byte-identical file is
    # always detected as unchanged, regardless of file type -- one hash function
    # works for both PDFs and markdown.
    # SJ - this checks the entire document at once, beofre chunking. 
    #This is how program knows if this is a new doc or existing document
    return hashlib.sha256(raw).hexdigest()


def _slugify(name: str) -> str:
    # Turns a filename into a safe primary key for the `documents`/`chunks`
    # tables, e.g. "BuySell-pws-get-invoice-....pdf" -> "buysell-pws-get-invoice-...".
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


_BUSINESS_MODEL_PATTERNS = [
    (re.compile(r"buy.?sell", re.IGNORECASE), "buysell", "BuySell"),
    (re.compile(r"nxm", re.IGNORECASE), "nxm", "NxM"),
]


def _detect_business_model(filename: str) -> tuple[str, str] | None:
    """Finds the business-model marker anywhere in the filename (not just a
    hyphen-delimited leading token, e.g. "BuySell-pws-..." and "NxMOverviewDeck"
    both need to match) and returns (category, canonical_label)."""
    for pattern, category, label in _BUSINESS_MODEL_PATTERNS:
        if pattern.search(filename):
            return category, label
    return None


# Per ADSK-BIZ-RULES.md: in the Buy Sell model, only Distributors can place
# orders to Autodesk -- Resellers cannot. These two manuals are entirely
# dedicated to the distributor-only APIs (unlike the general implementation
# guide, which covers all 5 Buy-Sell APIs together and can't be cleanly split
# at page level without also hiding legitimate reseller-facing content like
# GetOrderStatus/GetOrderDetails/GetInvoice). This is a real, document-level
# ACL boundary, not just narrative text -- a reseller session's retrieval
# query filters these out at the SQL level before anything reaches the LLM.
_DISTRIBUTOR_ONLY_FILES = {
    "BuySell-pws-get-myprice-service-reference-manual.pdf",
    "BuySell-pws-placeorder-v2-service-reference-manual.pdf",
}


def _pdf_acl_and_category(filename: str) -> tuple[list[str], str]:
    """Most PDFs are public reference documentation; a couple are restricted
    per real business rules (see _DISTRIBUTOR_ONLY_FILES above). The one
    other restricted file in this KB (a pricing spreadsheet) isn't a PDF and
    isn't ingested yet."""
    match = _detect_business_model(filename)
    category = match[0] if match else "other"
    acl = ["role:distributor"] if filename in _DISTRIBUTOR_ONLY_FILES else ["public"]
    return acl, category


_ACRONYMS = {"pws": "PWS", "api": "API", "v2": "v2", "v3": "v3"}


def _doc_title(filename: str) -> str:
    """Human-readable title derived from the filename, not the PDF's own text --
    the business-model identity (BuySell vs NxM) is a naming convention that
    never appears inside the document body itself (verified by inspection), so
    only the filename can supply it. See chunk_pdf's docstring for why this
    matters.

    The business-model marker is pulled out explicitly first (rather than
    relying purely on hyphen-splitting) because not every filename separates
    words with hyphens -- "NxMOverviewDeck.pdf" has no hyphens at all, and a
    naive split would produce "Nxmoverviewdeck", silently losing the exact
    signal this function exists to preserve.
    """
    stem = os.path.splitext(filename)[0]
    label = None
    for pattern, _, model_label in _BUSINESS_MODEL_PATTERNS:
        found = pattern.search(stem)
        if found:
            label = model_label
            stem = stem[: found.start()] + stem[found.end() :]
            break

    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)  # split camelCase boundaries
    words = [w for w in re.split(r"[-_\s]+", stem) if w]
    title_words = [_ACRONYMS.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(([label] if label else []) + title_words)


def _embed_in_batches(texts: list[str]) -> list[list[float]]:
    # OpenRouter (like most embedding APIs) has a per-request size limit, and a
    # 123-page PDF can produce well over 100 chunks -- batch instead of sending
    # everything in one call.
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(embed_texts(texts[i : i + EMBED_BATCH_SIZE]))
    return embeddings


def _load_chunks(path: str, filename: str, fallback_doc_id: str) -> list[Chunk]:
    # Dispatches to the right chunker by extension. Markdown carries its own
    # doc_id/acl/category/guaranteed via frontmatter; PDFs get theirs assigned
    # here from the filename convention, since a PDF has no frontmatter to read.
    if filename.endswith(".md"):
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        return chunk_markdown(raw, fallback_doc_id)

    if filename.endswith(".pdf"):
        acl, category = _pdf_acl_and_category(filename)
        pages = extract_pages(path)
        return chunk_pdf(pages, fallback_doc_id, acl, category, doc_title=_doc_title(filename))

    raise ValueError(f"unsupported file type: {filename}")


def ingest_corpus(corpus_dir: str = config.CORPUS_DIR) -> None:
    # One document at a time, each in its own transaction: hash-check -> skip
    # or (chunk -> embed -> delete old rows -> insert new rows) -> commit. A
    # failure partway through one document never leaves a half-updated doc
    # committed, and never blocks the rest of the corpus from ingesting.
    filenames = sorted(f for f in os.listdir(corpus_dir) if f.endswith((".md", ".pdf")))
    with psycopg.connect(config.DATABASE_URL, autocommit=False) as conn:
        for filename in filenames:
            path = os.path.join(corpus_dir, filename)
            with open(path, "rb") as fh:
                raw_bytes = fh.read()
            fallback_doc_id = _slugify(os.path.splitext(filename)[0])
            content_hash = _content_hash_bytes(raw_bytes)

            # Skip check: compare this file's hash against what's already
            # recorded for this doc_id. Nothing below runs (no chunking, no
            # embedding API calls, no writes) if the file hasn't changed.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content_hash FROM documents WHERE doc_id = %s",
                    (fallback_doc_id,),
                )
                row = cur.fetchone()

            if row and row[0] == content_hash:
                print(f"skip  (unchanged): {fallback_doc_id}")
                continue

            # New or changed file: chunk it, then embed every chunk's text.
            chunks = _load_chunks(path, filename, fallback_doc_id)
            if not chunks:
                print(f"warn  (no extractable text): {fallback_doc_id}")
                continue
            embeddings = _embed_in_batches([c.text for c in chunks])

            # Write phase: replace this doc's chunks atomically and record its
            # new hash, all in one transaction (committed below).
            with conn.cursor() as cur:
                # Delete first so a changed doc's stale chunks never linger in either index.
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (fallback_doc_id,))
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, content_hash, status, category, last_seen_at)
                    VALUES (%s, %s, 'active', %s, now())
                    ON CONFLICT (doc_id) DO UPDATE
                        SET content_hash = EXCLUDED.content_hash,
                            status = 'active',
                            category = EXCLUDED.category,
                            last_seen_at = now()
                    """,
                    (fallback_doc_id, content_hash, chunks[0].metadata.get("category")),
                )
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_id, doc_id, ordinal, heading, chunk_text, embedding, metadata, acl)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.ordinal,
                            chunk.heading,
                            chunk.text,
                            _vector_literal(embedding),
                            psycopg.types.json.Jsonb(chunk.metadata),
                            chunk.acl,
                        ),
                    )
            conn.commit()
            print(f"ingested: {fallback_doc_id} ({len(chunks)} chunks)")


if __name__ == "__main__":
    ingest_corpus()
