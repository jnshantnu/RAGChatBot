"""Hybrid retrieval: one chunks table, two index arms, ACL predicate inside each scan.

Both arms are parameterized (never string-interpolated) and filtered by
`acl && user_groups` so a restricted chunk can never enter the candidate set
for a session that lacks the matching group -- permission enforcement happens
before ranking, not after.
"""
from dataclasses import dataclass

import psycopg

# Over-fetch more rows than we'll ultimately use (TOP_K=5 in pipeline.py) so
# RRF fusion and reranking have a wide enough candidate pool to work with --
# fetching only 5 from each arm up front would silently cap recall before
# fusion even gets a chance to combine the two arms' rankings.
OVER_FETCH = 50


# One row from either search arm, before fusion. `score` means different
# things depending on which function produced it (see the comment on that field).
@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    heading: str
    chunk_text: str
    score: float  # arm-native score: ts_rank_cd (higher=better) or cosine distance (lower=better)
    metadata: dict


def keyword_search(conn: psycopg.Connection, query_text: str, user_groups: list[str], limit: int = OVER_FETCH) -> list[Candidate]:
    # plainto_tsquery ANDs every term together, so a single unmatched word (e.g.
    # "work" in "how does the downgrade policy work") zeroes out an otherwise
    # relevant chunk. Rewriting '&' -> '|' turns it into an OR-of-terms query,
    # closer to how a real keyword/BM25 arm behaves -- ts_rank_cd still rewards
    # chunks that match more terms. (True BM25 needs pg_search/VectorChord-BM25,
    # per the case study; this is Postgres's built-in approximation.)
    sql = """
        WITH q AS (
            SELECT to_tsquery('english', replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')) AS tsq
        )
        SELECT chunk_id, doc_id, heading, chunk_text, ts_rank_cd(tsv, q.tsq) AS score, metadata
        FROM chunks, q
        WHERE acl && %(groups)s
          AND index_version = 1
          AND tsv @@ q.tsq
        ORDER BY score DESC
        LIMIT %(limit)s
    """
    # This query hits the GIN index on `tsv` (see db/schema.sql). `%s` placeholders
    # keep every value parameterized -- query_text and user_groups are user-
    # influenced input, never string-interpolated into the SQL itself.
    with conn.cursor() as cur:
        cur.execute(sql, {"q": query_text, "groups": user_groups, "limit": limit})
        return [Candidate(*row) for row in cur.fetchall()]


def fetch_guaranteed_chunks(conn: psycopg.Connection, user_groups: list[str]) -> list[Candidate]:
    """Every guaranteed-inclusion chunk visible to this session, fetched directly --
    no keyword/semantic competition, no OVER_FETCH cutoff. Docs marked
    `guaranteed: true` (e.g. ADSK-BIZ-RULES.md) are small by design (a handful
    of chunks) and must never depend on out-competing the rest of a 1,000+
    chunk corpus for a spot in the top-20 candidates that make it to
    reranking -- a real regression we hit: "which apis i have access to"
    didn't surface the one chunk stating GetMyPrice/PlaceOrderV2 are
    distributor-only, because it didn't win a slot in that top 20 for this
    exact phrasing. Business rules that must always be considered can't be
    left to a competitive process that can silently drop them. `score` is a
    placeholder (0.0) -- these get reranked separately in retrieval/pipeline.py
    to judge relevance, not to decide inclusion.
    """
    sql = """
        SELECT chunk_id, doc_id, heading, chunk_text, 0.0 AS score, metadata
        FROM chunks
        WHERE acl && %(groups)s
          AND index_version = 1
          AND metadata->>'guaranteed' = 'true'
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"groups": user_groups})
        return [Candidate(*row) for row in cur.fetchall()]


def semantic_search(conn: psycopg.Connection, query_embedding: list[float], user_groups: list[str], limit: int = OVER_FETCH) -> list[Candidate]:
    # pgvector has no Python-native parameter type, so the embedding is
    # formatted as the text literal it expects and cast with `::vector` --
    # same trick as ingest/ingest.py's _vector_literal, used here for the
    # query side instead of the document side.
    vector_literal = "[" + ",".join(repr(v) for v in query_embedding) + "]"
    # `<=>` is pgvector's cosine-distance operator (lower = more similar); this
    # ORDER BY is what makes Postgres use the HNSW index on `embedding` instead
    # of scanning every row (see db/schema.sql).
    sql = """
        SELECT chunk_id, doc_id, heading, chunk_text,
               embedding <=> %(vec)s::vector AS distance, metadata
        FROM chunks
        WHERE acl && %(groups)s
          AND index_version = 1
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"vec": vector_literal, "groups": user_groups, "limit": limit})
        return [Candidate(*row) for row in cur.fetchall()]
