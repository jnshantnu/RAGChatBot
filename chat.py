"""End-to-end query pipeline: permission refusal -> rewrite -> retrieve -> gate -> generate.

Mirrors the two distinct "no" paths from the case study's architecture:
  - Permission refusal: decided BEFORE retrieval, from intent + role only.
    Never touches the corpus, so the request itself can't be used to probe
    what exists. Names the category, never a record.
  - Abstain: decided AFTER retrieval, from the confidence gate reading the
    post-fusion score spread -- not from permissions.

Query rewriting (retrieval/query_rewrite.py) sits between those two: it runs
on the ORIGINAL query (permission refusal already checked it), and its output
feeds both retrieval and generation. Role is still passed to generate_answer
separately too, even when a rewrite already mentions it -- the rewriter's
triggers are pattern-based and won't catch every phrasing, so the explicit
role stays as a general-purpose backstop rather than the only signal.
"""
import time
from dataclasses import asdict, dataclass, field

import psycopg

import config
from llm.generate import generate_answer
from retrieval.pipeline import retrieve
from retrieval.query_rewrite import rewrite_query
from retrieval.trace import TraceStep

# Keyword-triggered intent check for principal-only categories. A real system
# would use a small classifier or the query rewriter; a keyword match is
# enough to demonstrate the "decided before retrieval" shape at POC scale.
PERMISSION_RULES = [
    {
        "keywords": ["commission", "payout", "incentive", "margin"],
        "required_group": "role:principal",
        "category_name": "commission and incentive payout details",
        "restriction_label": "principal-level users",
    },
    {
        "keywords": ["placeorderv2", "place order api", "getmyprice", "get my price"],
        "required_group": "role:distributor",
        "category_name": "Place Order and GetMyPrice API details",
        "restriction_label": "distributor accounts",
    },
]


# Everything the FastAPI endpoint (app/main.py) returns to the client, one
# instance per request. Fields default to the "nothing interesting happened"
# case so early-return paths (refusal, abstain) don't need to set every field.
@dataclass
class ChatResponse:
    query: str
    role: str
    partner: str
    answer: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    permission_refused: bool = False
    degraded_rerank: bool = True
    scores: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    rewritten_query: str | None = None  # set only when query_rewrite.py actually changed something
    rewrite_rules_applied: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)  # debug-mode flowchart data, see retrieval/trace.py


def _user_groups(role: str, partner: str) -> list[str]:
    # Translates a UI role/partner selection into the ACL group list used to
    # filter retrieval (see retrieval/hybrid_search.py's `acl && :groups`
    # predicate). "public" is always included; role adds at most one more group.
    groups = ["public", f"partner:{partner}"]
    if role == "principal":
        groups.append("role:principal")
    elif role == "distributor":
        groups.append("role:distributor")
    return groups


def _visible_scores(results) -> list[dict]:
    """Debug/UI score payload -- everything retrieved is citable now, so
    nothing is filtered out here beyond capping the length for display."""
    return [asdict(r) for r in results][:5]


def _check_permission_refusal(query: str, user_groups: list[str]) -> str | None:
    # Simple substring match against each rule's keyword list -- if the query
    # mentions a restricted topic AND the session lacks the required group,
    # return a refusal message immediately. Returns None (no rule fired) to
    # let the caller fall through to normal retrieval.
    lowered = query.lower()
    for rule in PERMISSION_RULES:
        if any(kw in lowered for kw in rule["keywords"]) and rule["required_group"] not in user_groups:
            return (
                f"I can't share {rule['category_name']} with this account type -- "
                f"that information is restricted to {rule['restriction_label']}."
            )
    return None


def answer_query(query: str, role: str, partner: str) -> ChatResponse:
    # This is the one function app/main.py calls per request. Three possible
    # exit points, in order: permission refusal (no corpus access at all),
    # confidence-gate abstain (retrieved but not confident enough), or a real
    # generated answer -- each returns its own ChatResponse immediately.
    t_start = time.perf_counter()
    user_groups = _user_groups(role, partner)

    # Exit 1: refuse before touching the corpus at all, if the query names a
    # restricted category the session's role doesn't have.
    refusal = _check_permission_refusal(query, user_groups)
    if refusal:
        return ChatResponse(
            query=query, role=role, partner=partner,
            answer=refusal, permission_refused=True,
            timings_ms={"total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
        )

    # Rewrite (retrieval/query_rewrite.py): role injection, synonym
    # normalization, or bare-term expansion, whichever rule's trigger
    # matches. A no-op for queries that don't match any trigger -- retrieval
    # and generation below just get the original query text back unchanged.
    t0 = time.perf_counter()
    rewrite = rewrite_query(query, role)
    retrieval_query = rewrite.rewritten_query
    rewrite_trace = TraceStep(
        name="Query Rewrite", inputs={"query": query, "role": role},
        outputs={"rewritten_query": retrieval_query, "rules_applied": rewrite.rules_applied},
        timing_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

    with psycopg.connect(config.DATABASE_URL) as conn:
        result = retrieve(conn, retrieval_query, user_groups)

        # Exit 2: retrieval ran, but the confidence gate says nothing found is
        # relevant enough to answer from -- skip the LLM call entirely.
        if result.gate.abstain:
            return ChatResponse(
                query=query, role=role, partner=partner,
                answer="I don't have that information.",
                abstained=True,
                degraded_rerank=result.degraded_rerank,
                scores=_visible_scores(result.candidates),
                rewritten_query=rewrite.rewritten_query if rewrite.changed else None,
                rewrite_rules_applied=rewrite.rules_applied,
                trace=[rewrite_trace, *result.trace],
                timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
            )

        # Exit 3: generate a real, cited answer. result.top_k already includes
        # every guaranteed chunk (e.g. ADSK-BIZ-RULES.md) alongside the
        # competitive top-K -- all of it is citable, see llm/generate.py. The
        # role is passed through explicitly too (see module docstring) so the
        # model can resolve "me" even when the rewriter's triggers didn't fire.
        t0 = time.perf_counter()
        generated = generate_answer(retrieval_query, result.top_k, role)
        t_generate = round((time.perf_counter() - t0) * 1000, 1)
        result.timings_ms["generate_ms"] = t_generate
        generate_trace = TraceStep(
            name="Generate Answer",
            inputs={"role": role, "chunks_in_context": len(result.top_k)},
            outputs={
                "answer_chars": len(generated.answer),
                "citations": generated.cited_chunk_ids,
                "uncited_claims_flagged": generated.uncited_claims_flagged,
            },
            timing_ms=t_generate,
            detail={"answer": generated.answer},
        )

    return ChatResponse(
        query=query, role=role, partner=partner,
        answer=generated.answer,
        citations=generated.cited_chunk_ids,
        degraded_rerank=result.degraded_rerank,
        scores=_visible_scores(result.top_k),
        rewritten_query=rewrite.rewritten_query if rewrite.changed else None,
        rewrite_rules_applied=rewrite.rules_applied,
        trace=[rewrite_trace, *result.trace, generate_trace],
        timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
    )
