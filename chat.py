"""End-to-end query pipeline: permission refusal -> retrieve -> gate -> generate.

Mirrors the two distinct "no" paths from the case study's architecture:
  - Permission refusal: decided BEFORE retrieval, from intent + role only.
    Never touches the corpus, so the request itself can't be used to probe
    what exists. Names the category, never a record.
  - Abstain: decided AFTER retrieval, from the confidence gate reading the
    post-fusion score spread -- not from permissions.
"""
import time
from dataclasses import asdict, dataclass, field

import psycopg

import config
from llm.generate import generate_answer
from retrieval.pipeline import retrieve

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
    internal_guidance_used: bool = False  # non-revealing flag: some internal doc informed this
    # answer's content, but never its identity or text -- see llm/generate.py


def _user_groups(role: str, partner: str) -> list[str]:
    groups = ["public", f"partner:{partner}"]
    if role == "principal":
        groups.append("role:principal")
    elif role == "distributor":
        groups.append("role:distributor")
    return groups


def _visible_scores(results) -> list[dict]:
    """Debug/UI score payload -- internal chunks are never included, so an
    internal-only doc can influence an answer's content without its existence
    ever surfacing in the API response or the web UI's score panel."""
    return [asdict(r) for r in results if not r.is_internal][:5]


def _check_permission_refusal(query: str, user_groups: list[str]) -> str | None:
    lowered = query.lower()
    for rule in PERMISSION_RULES:
        if any(kw in lowered for kw in rule["keywords"]) and rule["required_group"] not in user_groups:
            return (
                f"I can't share {rule['category_name']} with this account type -- "
                f"that information is restricted to {rule['restriction_label']}."
            )
    return None


def answer_query(query: str, role: str, partner: str) -> ChatResponse:
    t_start = time.perf_counter()
    user_groups = _user_groups(role, partner)

    refusal = _check_permission_refusal(query, user_groups)
    if refusal:
        return ChatResponse(
            query=query, role=role, partner=partner,
            answer=refusal, permission_refused=True,
            timings_ms={"total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
        )

    with psycopg.connect(config.DATABASE_URL) as conn:
        result = retrieve(conn, query, user_groups)

        if result.gate.abstain:
            return ChatResponse(
                query=query, role=role, partner=partner,
                answer="I don't have that information.",
                abstained=True,
                degraded_rerank=result.degraded_rerank,
                scores=_visible_scores(result.candidates),
                timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
            )

        citable = [r for r in result.top_k if not r.is_internal]
        internal = [r for r in result.top_k if r.is_internal]

        t0 = time.perf_counter()
        generated = generate_answer(query, citable, internal)
        result.timings_ms["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return ChatResponse(
        query=query, role=role, partner=partner,
        answer=generated.answer,
        citations=generated.cited_chunk_ids,
        degraded_rerank=result.degraded_rerank,
        scores=_visible_scores(result.top_k),
        internal_guidance_used=bool(internal),
        timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
    )
