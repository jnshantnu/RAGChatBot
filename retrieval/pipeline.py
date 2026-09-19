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

Two modes, one retrieval logic: `retrieve(..., mode=...)` dispatches to
`_retrieve_sequential` (the original behavior, "Baseline") or
`_retrieve_parallel` ("Optimized"). Both call the same underlying primitives
(embed_query, keyword_search, semantic_search, fetch_guaranteed_chunks,
reciprocal_rank_fusion, rerank, confidence_gate) with the same thresholds.
Optimized differs in two ways, and they are not equally valuable -- measured
over paired runs, not assumed:

  1. Scheduling: the independent steps run concurrently and the two rerank
     calls become one batched call. This produced NO reliable speedup
     (median 0.87x over 18 paired runs, faster in only 4): the overlappable
     work is tens of milliseconds, and the real critical path -- CPU-bound
     reranking, then LLM generation -- is serial. It is kept because it is
     verified behavior-identical, but it is not where the gain comes from.
  2. Reranker input length (TUNED_MAX_LENGTH in rerank.py): this is where the
     measured ~2x reranking speedup comes from, and unlike scheduling it is a
     real trade-off -- it changes what text the cross-encoder sees, so it can
     change which chunks rank first. See rerank.py for the validation and cost.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import psycopg

import config
from ingest.embed import embed_query
from retrieval.confidence import MIN_RERANK_SCORE, GateResult, confidence_gate
from retrieval.hybrid_search import fetch_guaranteed_chunks, keyword_search, semantic_search
from retrieval.rerank import DEFAULT_MAX_LENGTH, RERANK_CANDIDATES, TUNED_MAX_LENGTH, rerank
from retrieval.rrf import FusedResult, reciprocal_rank_fusion
from retrieval.trace import TraceStep, sort_trace, summarize_candidates

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
    # Debug-mode flowchart data -- see retrieval/trace.py and app/streamlit_app.py.
    trace: list[TraceStep] = field(default_factory=list)


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


def retrieve(conn: psycopg.Connection, query: str, user_groups: list[str], mode: str = "sequential") -> RetrievalResult:
    if mode == "parallel":
        return _retrieve_parallel(conn, query, user_groups)
    return _retrieve_sequential(conn, query, user_groups)


def _retrieve_sequential(conn: psycopg.Connection, query: str, user_groups: list[str]) -> RetrievalResult:
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
    trace: list[TraceStep] = []

    t0 = time.perf_counter()
    query_embedding = embed_query(query)
    t_embed = round((time.perf_counter() - t0) * 1000, 1)
    timings["embed_ms"] = t_embed
    trace.append(TraceStep(
        name="Embed Query", inputs={"query": query}, timing_ms=t_embed, started_at=t0,
        outputs={"embedding_dims": len(query_embedding)},
    ))

    t0 = time.perf_counter()
    keyword_results = keyword_search(conn, query, user_groups)
    t_kw = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        name="Keyword Search", inputs={"query": query, "user_groups": user_groups}, timing_ms=t_kw, started_at=t0,
        outputs={"candidates_found": len(keyword_results)},
        detail=summarize_candidates(keyword_results, "score"),
    ))

    t0 = time.perf_counter()
    semantic_results = semantic_search(conn, query_embedding, user_groups)
    t_sem = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        name="Semantic Search", inputs={"embedding_dims": len(query_embedding), "user_groups": user_groups}, timing_ms=t_sem,
        started_at=t0, outputs={"candidates_found": len(semantic_results)},
        detail=summarize_candidates(semantic_results, "score"),
    ))
    timings["retrieve_ms"] = round(t_kw + t_sem, 1)

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(keyword_results, semantic_results)
    timings["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        name="RRF Fusion",
        inputs={"keyword_candidates": len(keyword_results), "semantic_candidates": len(semantic_results)},
        outputs={"fused_candidates": len(fused), "top_fused_score": round(fused[0].fused_score, 4) if fused else 0.0},
        timing_ms=timings["fuse_ms"], started_at=t0,
        detail=summarize_candidates(fused, "fused_score"),
    ))

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
    trace.append(TraceStep(
        name="Rerank (competitive)",
        # `candidates_received` is exactly what step 5 (RRF Fusion) handed to
        # this step -- the top RERANK_CANDIDATES of the fused list, in their
        # pre-rerank order/fused_score -- so this step is readable on its own
        # without cross-referencing step 5's own detail panel.
        inputs={
            "query": query,
            "candidates_received": summarize_candidates(fused[:RERANK_CANDIDATES], "fused_score"),
        },
        outputs={
            # RERANK_CANDIDATES (20) is the hard cap on how many chunks the
            # cross-encoder ever scores per query, regardless of how many
            # came out of RRF Fusion -- see retrieval/rerank.py.
            "candidates_reranked": len(ranked),
            "degraded": degraded_rerank,
            "top_rerank_score": round(ranked[0].rerank_score, 4) if ranked and ranked[0].rerank_score is not None else None,
            "reranker_reads_tokens": DEFAULT_MAX_LENGTH,
        },
        timing_ms=timings["rerank_ms"], started_at=t0,
        detail=summarize_candidates(ranked, "rerank_score"),
    ))

    # Guaranteed-inclusion path: fetch every guaranteed chunk directly (never
    # subject to the top-20 rerank cutoff above), then rerank just this small
    # set on its own to judge relevance -- not to decide inclusion, all of
    # them are included regardless, only to let the gate know whether any of
    # them are actually relevant to this specific question.
    t0 = time.perf_counter()
    guaranteed_candidates = fetch_guaranteed_chunks(conn, user_groups)
    t_fetch = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        name="Guaranteed Fetch", inputs={"user_groups": user_groups}, timing_ms=t_fetch, started_at=t0,
        outputs={"guaranteed_chunks_found": len(guaranteed_candidates)},
        detail=[{"doc_id": c.doc_id, "heading": c.heading} for c in guaranteed_candidates],
    ))

    t0 = time.perf_counter()
    guaranteed_fused =[_as_fused(c, "guaranteed") for c in guaranteed_candidates]
    guaranteed_received = [{"doc_id": c.doc_id, "heading": c.heading} for c in guaranteed_fused]
    best_guaranteed_score = 0.0
    if guaranteed_fused and not degraded_rerank:
        try:
            guaranteed_fused = rerank(query, guaranteed_fused, top_n=len(guaranteed_fused))
            best_guaranteed_score = guaranteed_fused[0].rerank_score or 0.0
        except Exception:
            logger.exception("guaranteed-chunk rerank failed; including unscored")
    t_grerank = round((time.perf_counter() - t0) * 1000, 1)
    timings["guaranteed_ms"] = round(t_fetch + t_grerank, 1)
    trace.append(TraceStep(
        # `candidates_received` mirrors step 7 (Guaranteed Fetch)'s own
        # output -- shown again here so this step is self-contained too.
        name="Rerank (guaranteed)", inputs={"query": query, "candidates_received": guaranteed_received},
        outputs={"best_guaranteed_score": round(best_guaranteed_score, 4), "reranker_reads_tokens": DEFAULT_MAX_LENGTH},
        timing_ms=t_grerank, started_at=t0,
        detail=summarize_candidates(guaranteed_fused, "rerank_score"),
    ))

    t0 = time.perf_counter()
    gate = confidence_gate(ranked, semantic_results, keyword_results, reranked=not degraded_rerank)
    if gate.abstain and best_guaranteed_score >= MIN_RERANK_SCORE:
        # A strong guaranteed-chunk match rescues the gate exactly like a
        # strong competitively-retrieved match would -- e.g. a question
        # answerable only from ADSK-BIZ-RULES.md shouldn't abstain just
        # because nothing else matched.
        gate.abstain = False
        gate.reason = "ok_guaranteed_content"
        gate.best_score = max(gate.best_score, best_guaranteed_score)
    trace.append(TraceStep(
        name="Confidence Gate",
        inputs={
            "top_competitive_score": round(ranked[0].rerank_score, 4) if ranked and ranked[0].rerank_score is not None else None,
            "best_guaranteed_score": round(best_guaranteed_score, 4),
        },
        outputs={"abstain": gate.abstain, "reason": gate.reason, "best_score": round(gate.best_score, 4)},
        timing_ms=round((time.perf_counter() - t0) * 1000, 1), started_at=t0,
    ))

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
        trace=trace,
    )


def _retrieve_parallel(conn: psycopg.Connection, query: str, user_groups: list[str]) -> RetrievalResult:
    """The "Optimized" mode. Scheduling differs from _retrieve_sequential:
    keyword search, embed+semantic search, and guaranteed-fetch all run
    concurrently instead of one after another (none of them actually depend
    on each other -- keyword search only needs the query text, and guaranteed
    fetch only needs the ACL groups), and the two rerank calls (competitive +
    guaranteed) are merged into one batched cross-encoder call instead of two,
    since a cross-encoder scores each (query, chunk) pair independently
    regardless of what else is in its batch -- merging changes nothing about
    the resulting scores, only how many times model.predict() gets invoked.
    Those two changes are behavior-identical to sequential mode (see the
    module docstring for how little speed they buy).

    The one deliberate behavioral difference is the reranker's input length
    (TUNED_MAX_LENGTH): that is what actually makes this mode faster, and it
    can change which chunks rank first -- validated, and its cost documented,
    in rerank.py.

    Each concurrent DB-bound branch opens its own short-lived connection
    rather than sharing `conn` across threads -- psycopg connections aren't
    safe for concurrent use from multiple threads at once.
    """
    timings: dict = {}

    def _run_semantic():
        t0 = time.perf_counter()
        embedding = embed_query(query)
        t_embed = round((time.perf_counter() - t0) * 1000, 1)
        embed_step = TraceStep(
            name="Embed Query", inputs={"query": query}, timing_ms=t_embed, started_at=t0,
            outputs={"embedding_dims": len(embedding)},
        )
        t0 = time.perf_counter()
        with psycopg.connect(config.DATABASE_URL) as local_conn:
            semantic_results = semantic_search(local_conn, embedding, user_groups)
        t_sem = round((time.perf_counter() - t0) * 1000, 1)
        semantic_step = TraceStep(
            name="Semantic Search",
            inputs={"embedding_dims": len(embedding), "user_groups": user_groups}, timing_ms=t_sem, started_at=t0,
            outputs={"candidates_found": len(semantic_results)},
            detail=summarize_candidates(semantic_results, "score"),
        )
        return semantic_results, t_embed, t_sem, embed_step, semantic_step

    def _run_keyword():
        t0 = time.perf_counter()
        with psycopg.connect(config.DATABASE_URL) as local_conn:
            keyword_results = keyword_search(local_conn, query, user_groups)
        t_kw = round((time.perf_counter() - t0) * 1000, 1)
        step = TraceStep(
            name="Keyword Search", inputs={"query": query, "user_groups": user_groups}, timing_ms=t_kw, started_at=t0,
            outputs={"candidates_found": len(keyword_results)},
            detail=summarize_candidates(keyword_results, "score"),
        )
        return keyword_results, t_kw, step

    def _run_guaranteed_fetch():
        t0 = time.perf_counter()
        with psycopg.connect(config.DATABASE_URL) as local_conn:
            guaranteed_candidates = fetch_guaranteed_chunks(local_conn, user_groups)
        t_fetch = round((time.perf_counter() - t0) * 1000, 1)
        step = TraceStep(
            name="Guaranteed Fetch", inputs={"user_groups": user_groups}, timing_ms=t_fetch, started_at=t0,
            outputs={"guaranteed_chunks_found": len(guaranteed_candidates)},
            detail=[{"doc_id": c.doc_id, "heading": c.heading} for c in guaranteed_candidates],
        )
        return guaranteed_candidates, t_fetch, step

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_semantic = executor.submit(_run_semantic)
        future_keyword = executor.submit(_run_keyword)
        future_guaranteed = executor.submit(_run_guaranteed_fetch)

        semantic_results, t_embed, t_sem, embed_step, semantic_step = future_semantic.result()
        keyword_results, t_kw, keyword_step = future_keyword.result()
        guaranteed_candidates, t_fetch, guaranteed_fetch_step = future_guaranteed.result()

    timings["embed_ms"] = t_embed
    timings["retrieve_ms"] = round(t_kw + t_sem, 1)
    trace = [embed_step, keyword_step, semantic_step, guaranteed_fetch_step]

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(keyword_results, semantic_results)
    timings["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        name="RRF Fusion",
        inputs={"keyword_candidates": len(keyword_results), "semantic_candidates": len(semantic_results)},
        outputs={"fused_candidates": len(fused), "top_fused_score": round(fused[0].fused_score, 4) if fused else 0.0},
        timing_ms=timings["fuse_ms"], started_at=t0,
        detail=summarize_candidates(fused, "fused_score"),
    ))

    # Batched rerank: one cross-encoder call scores both the competitive
    # slice and the guaranteed chunks together, instead of two separate
    # calls. Deliberately NOT de-duplicated before batching: sequential mode
    # reranks its competitive slice as-is (a guaranteed chunk that also won a
    # competitive spot stays in that list, uncut, and can drive the gate's
    # main/dominance checks from there) and separately reranks the
    # guaranteed set as its own, independent pass -- so this merges those
    # same two lists into one batch without changing either one's contents.
    # Provenance is tracked by object identity (id()), not by the
    # `is_guaranteed` chunk-metadata flag: a chunk that's both guaranteed AND
    # independently won a competitive spot appears here as two *separate*
    # FusedResult objects (one from each source list), both flagged
    # is_guaranteed=True by metadata -- only object identity can tell them
    # apart, which is what lets this exactly reproduce sequential mode's two
    # independent rerank passes while still only calling the model once.
    t0 = time.perf_counter()
    degraded_rerank = False
    competitive_slice = fused[:RERANK_CANDIDATES]
    guaranteed_fused_raw = [_as_fused(c, "guaranteed") for c in guaranteed_candidates]
    competitive_object_ids = {id(c) for c in competitive_slice}

    merged = competitive_slice + guaranteed_fused_raw
    scored = merged
    if merged:
        try:
            scored = rerank(query, merged, top_n=len(merged), max_length=TUNED_MAX_LENGTH)
        except Exception:
            logger.exception("batched rerank failed, falling back to RRF order")
            degraded_rerank = True
    t_rerank = round((time.perf_counter() - t0) * 1000, 1)
    timings["rerank_ms"] = t_rerank
    timings["guaranteed_ms"] = t_fetch  # rerank cost is folded into rerank_ms above -- batched, not separate

    # competitive_scored mirrors sequential's `ranked` exactly (same input
    # slice, same model, same resulting scores/order); guaranteed_scored
    # mirrors sequential's post-rerank `guaranteed_fused`.
    competitive_scored = [r for r in scored if id(r) in competitive_object_ids]
    guaranteed_scored = [r for r in scored if id(r) not in competitive_object_ids]
    top_competitive_score = (
        competitive_scored[0].rerank_score
        if competitive_scored and competitive_scored[0].rerank_score is not None
        else None
    )
    best_guaranteed_score = (guaranteed_scored[0].rerank_score or 0.0) if guaranteed_scored and not degraded_rerank else 0.0

    # Both rerank nodes carry `batched_call: True` and the same started_at/timing
    # -- the debug UI keys off that to mark them as one shared model call.
    trace.append(TraceStep(
        name="Rerank (competitive)",
        inputs={"query": query, "candidates_received": summarize_candidates(competitive_slice, "fused_score")},
        outputs={
            "candidates_reranked": len(competitive_slice),
            "degraded": degraded_rerank,
            "top_rerank_score": round(top_competitive_score, 4) if top_competitive_score is not None else None,
            "batched_call": True,
            "reranker_reads_tokens": TUNED_MAX_LENGTH,
        },
        timing_ms=t_rerank, started_at=t0,
        detail=summarize_candidates(competitive_scored, "rerank_score"),
    ))
    trace.append(TraceStep(
        name="Rerank (guaranteed)",
        inputs={"query": query, "candidates_received": [{"doc_id": c.doc_id, "heading": c.heading} for c in guaranteed_fused_raw]},
        outputs={
            "best_guaranteed_score": round(best_guaranteed_score, 4),
            "batched_call": True,
            "reranker_reads_tokens": TUNED_MAX_LENGTH,
        },
        # Same batched call as "Rerank (competitive)" above -- timing shown
        # on both cards (not split or zeroed) since that's honestly when
        # this scoring actually happened, not a separate cost on top of it.
        timing_ms=t_rerank, started_at=t0,
        detail=summarize_candidates(guaranteed_scored, "rerank_score"),
    ))

    t0 = time.perf_counter()
    gate_input = competitive_scored if not degraded_rerank else fused
    gate = confidence_gate(gate_input, semantic_results, keyword_results, reranked=not degraded_rerank)
    if gate.abstain and best_guaranteed_score >= MIN_RERANK_SCORE:
        gate.abstain = False
        gate.reason = "ok_guaranteed_content"
        gate.best_score = max(gate.best_score, best_guaranteed_score)
    trace.append(TraceStep(
        name="Confidence Gate",
        inputs={
            "top_competitive_score": round(top_competitive_score, 4) if top_competitive_score is not None else None,
            "best_guaranteed_score": round(best_guaranteed_score, 4),
        },
        outputs={"abstain": gate.abstain, "reason": gate.reason, "best_score": round(gate.best_score, 4)},
        timing_ms=round((time.perf_counter() - t0) * 1000, 1), started_at=t0,
    ))

    # Matches sequential mode's post-gate dedup: a chunk that's both
    # guaranteed AND independently won a competitive spot must not also take
    # one of the competitive top-K slots -- it's already unconditionally
    # included via guaranteed_scored, so double-including it here would
    # duplicate its text in the final prompt/citations.
    competitive_for_topk = [r for r in competitive_scored if not r.is_guaranteed]
    top_k = competitive_for_topk[:TOP_K] + guaranteed_scored

    return RetrievalResult(
        query=query,
        user_groups=user_groups,
        candidates=competitive_scored if not degraded_rerank else fused,
        top_k=top_k,
        gate=gate,
        degraded_rerank=degraded_rerank,
        best_guaranteed_score=best_guaranteed_score,
        timings_ms=timings,
        trace=sort_trace(trace),
    )
