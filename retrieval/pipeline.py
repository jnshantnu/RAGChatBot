"""Retriever deep-dive, wired together: query -> embed -> hybrid search -> RRF -> rerank -> gate.

If the cross-encoder can't be loaded, results stay in RRF order and the
response is flagged `degraded.rerank=True` rather than silently pretending
reranking happened -- matching the case study's own dashed degraded path.

Guaranteed-inclusion chunks (e.g. ADSK-BIZ-RULES.md, marked `guaranteed: true`)
are handled on a completely separate path -- see fetch_guaranteed_chunks in
hybrid_search.py for why: a 5-chunk doc competing for one of 20 rerank slots
against a 1,000+ chunk corpus can lose that competition on a given phrasing
and vanish from the answer entirely, even though its content is supposed to
shape every relevant answer. So guaranteed content is always fetched in full,
reranked in its own small pass (to judge relevance, not to decide inclusion),
and always handed to generation -- fully citable, same as anything else --
while still letting a strong guaranteed match rescue the confidence gate, the
same way a strong competitively-retrieved match would.
"""
import logging
import time
from dataclasses import dataclass, field

import psycopg

from ingest.embed import embed_query
from retrieval.confidence import MIN_RERANK_SCORE, GateResult, confidence_gate
from retrieval.hybrid_search import fetch_guaranteed_chunks, keyword_search, semantic_search
from retrieval.rerank import rerank
from retrieval.rrf import FusedResult, reciprocal_rank_fusion

TOP_K = 5

logger = logging.getLogger(__name__)


# Everything one retrieve() call produces. `candidates` is the full reranked
# competitive list (used by eval/run_eval.py and retrieval/cli_test.py);
# `top_k` is the smaller slice chat.py actually hands to the LLM, and always
# includes every guaranteed chunk regardless of how it did in that competition.
@dataclass
class RetrievalResult:
    query: str
    user_groups: list[str]
    candidates: list[FusedResult]
    top_k: list[FusedResult]
    gate: GateResult
    degraded_rerank: bool
    timings_ms: dict = field(default_factory=dict)
    # Best rerank score among guaranteed chunks -- NOT whether any are present
    # in top_k (they always are, since inclusion is unconditional). Used by
    # the confidence gate's rescue check: "was this genuinely relevant" rather
    # than "was it merely attached to the prompt."
    best_guaranteed_score: float = 0.0


def _as_fused(candidate, arm_label: str) -> FusedResult:
    return FusedResult(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        heading=candidate.heading,
        chunk_text=candidate.chunk_text,
        fused_score=0.0,
        arms=[arm_label],
        metadata=candidate.metadata,
    )


def retrieve(conn: psycopg.Connection, query: str, user_groups: list[str]) -> RetrievalResult:
    # The full pipeline for one query, in order:
    #   1. embed the query text
    #   2. run both search arms (keyword + semantic) in parallel SQL queries
    #   3. fuse their rankings (RRF)
    #   4. rerank the fused list with a cross-encoder for real relevance scores
    #   5. separately fetch + rerank ALL guaranteed-inclusion chunks (not competitive)
    #   6. run the confidence gate, letting either a strong competitive or strong guaranteed match pass it
    #   7. pick the final top-K to hand back to chat.py
    # Each stage's wall-clock time is recorded in `timings` for the UI's
    # timings panel and for spotting which stage is slow.
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

    # Rerank the fused list with the cross-encoder (retrieval/rerank.py). If the
    # model fails to load or errors out, fall back to RRF order rather than
    # crashing the whole request -- but flag it as degraded so downstream code
    # (the confidence gate, the UI) knows the scores aren't the real thing.
    t0 = time.perf_counter()
    degraded_rerank = False
    ranked = fused
    try:
        ranked = rerank(query, fused)
    except Exception:
        logger.exception("rerank failed, falling back to RRF order")
        degraded_rerank = True
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Guaranteed-inclusion path: fetch every guaranteed chunk directly (never
    # subject to the top-20 rerank cutoff above), then rerank just this small
    # set on its own to judge relevance -- not to decide inclusion, all of
    # them are included regardless, only to let the gate know whether any of
    # them are actually relevant to this specific question.
    t0 = time.perf_counter()
    guaranteed_candidates = fetch_guaranteed_chunks(conn, user_groups)
    guaranteed_fused = [_as_fused(c, "guaranteed") for c in guaranteed_candidates]
    best_guaranteed_score = 0.0
    if guaranteed_fused and not degraded_rerank:
        try:
            guaranteed_fused = rerank(query, guaranteed_fused, top_n=len(guaranteed_fused))
            best_guaranteed_score = guaranteed_fused[0].rerank_score or 0.0
        except Exception:
            logger.exception("guaranteed-chunk rerank failed; including unscored")
    timings["guaranteed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    gate = confidence_gate(ranked, semantic_results, keyword_results, reranked=not degraded_rerank)
    if gate.abstain and best_guaranteed_score >= MIN_RERANK_SCORE:
        # A strong guaranteed-chunk match rescues the gate exactly like a
        # strong competitively-retrieved match would -- e.g. a question
        # answerable only from ADSK-BIZ-RULES.md shouldn't abstain just
        # because nothing else matched.
        gate.abstain = False
        gate.reason = "ok_guaranteed_content"
        gate.best_score = max(gate.best_score, best_guaranteed_score)

    # The competitive pipeline still fills its own top-K slots; guaranteed
    # content is no longer drawn from that competition at all -- every
    # guaranteed chunk found above is always included, in addition to it.
    competitive_ranked = [r for r in ranked if not r.is_guaranteed]
    top_k = competitive_ranked[:TOP_K] + guaranteed_fused

    return RetrievalResult(
        query=query,
        user_groups=user_groups,
        candidates=ranked,
        top_k=top_k,
        gate=gate,
        degraded_rerank=degraded_rerank,
        best_guaranteed_score=best_guaranteed_score,
        timings_ms=timings,
    )
