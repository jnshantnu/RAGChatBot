"""Retriever deep-dive, wired together: query -> embed -> hybrid search -> RRF -> rerank -> gate.

If the cross-encoder can't be loaded, results stay in RRF order and the
response is flagged `degraded.rerank=True` rather than silently pretending
reranking happened -- matching the case study's own dashed degraded path.
"""
import logging
import time
from dataclasses import dataclass, field

import psycopg

from ingest.embed import embed_query
from retrieval.confidence import GateResult, confidence_gate
from retrieval.hybrid_search import keyword_search, semantic_search
from retrieval.rerank import rerank
from retrieval.rrf import FusedResult, reciprocal_rank_fusion

TOP_K = 5

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    user_groups: list[str]
    candidates: list[FusedResult]
    top_k: list[FusedResult]
    gate: GateResult
    degraded_rerank: bool
    timings_ms: dict = field(default_factory=dict)


def retrieve(conn: psycopg.Connection, query: str, user_groups: list[str]) -> RetrievalResult:
    timings = {}

    t0 = time.perf_counter()
    query_embedding = embed_query(query)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    keyword_results = keyword_search(conn, query, user_groups)
    semantic_results = semantic_search(conn, query_embedding, user_groups)
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(keyword_results, semantic_results)
    timings["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    degraded_rerank = False
    ranked = fused
    try:
        ranked = rerank(query, fused)
    except Exception:
        logger.exception("rerank failed, falling back to RRF order")
        degraded_rerank = True
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    gate = confidence_gate(ranked, semantic_results, keyword_results, reranked=not degraded_rerank)

    # Citable and internal chunks get their own top-K slots rather than sharing
    # one slice of `ranked` -- otherwise a few dominant internal matches (e.g.
    # ADSK-BIZ-RULES.md scoring 0.99) crowd out real citable content, leaving
    # only near-zero-score filler in what's shown to the user as "top sources".
    citable_ranked = [r for r in ranked if not r.is_internal]
    internal_ranked = [r for r in ranked if r.is_internal]
    top_k = citable_ranked[:TOP_K] + internal_ranked[:TOP_K]

    return RetrievalResult(
        query=query,
        user_groups=user_groups,
        candidates=ranked,
        top_k=top_k,
        gate=gate,
        degraded_rerank=degraded_rerank,
        timings_ms=timings,
    )
