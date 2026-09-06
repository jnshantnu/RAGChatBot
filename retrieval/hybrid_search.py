"""Hybrid retrieval: one chunks table, two index arms, ACL predicate inside each scan.

Both arms are parameterized (never string-interpolated) and filtered by
`acl && user_groups` so a restricted chunk can never enter the candidate set
for a session that lacks the matching group -- permission enforcement happens
before ranking, not after.
"""
from dataclasses import dataclass

import psycopg

OVER_FETCH = 50


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
    with conn.cursor() as cur:
        cur.execute(sql, {"q": query_text, "groups": user_groups, "limit": limit})
        return [Candidate(*row) for row in cur.fetchall()]


def semantic_search(conn: psycopg.Connection, query_embedding: list[float], user_groups: list[str], limit: int = OVER_FETCH) -> list[Candidate]:
    vector_literal = "[" + ",".join(repr(v) for v in query_embedding) + "]"
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
