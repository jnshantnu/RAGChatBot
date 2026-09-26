"""Structured per-step trace: what went into and came out of each retrieval/
generation stage for one query. Captured unconditionally in
retrieval/pipeline.py and chat.py -- cheap (a handful of summary dicts, no
extra API/DB calls), the same way timings_ms already is -- and only rendered
when the Streamlit UI's debug-mode toggle is on (app/streamlit_app.py).
"""
from dataclasses import dataclass, field

# Canonical step order for the debug-mode flowchart -- shared by
# retrieval/pipeline.py (which sorts its trace into this order before
# returning, regardless of which order threads finished in under
# mode="parallel") and app/streamlit_app.py (which uses it to number and
# position nodes). One shared list so the two can never drift apart.
PIPELINE_STEP_ORDER = [
    "Query Rewrite", "Embed Query", "Keyword Search", "Semantic Search", "RRF Fusion",
    "Rerank (competitive)", "Guaranteed Fetch", "Rerank (guaranteed)", "Confidence Gate",
    "LLM Query Rewrite Retry", "Generate Answer",
]


def sort_trace(trace: list) -> list:
    # Parallel execution finishes steps in whatever order threads complete,
    # not pipeline order -- this restores canonical order so the debug UI
    # always reads top-to-bottom the same way regardless of execution mode.
    order = {name: i for i, name in enumerate(PIPELINE_STEP_ORDER)}
    return sorted(trace, key=lambda s: order.get(s.name, len(order)))


@dataclass
class TraceStep:
    name: str            # matches a node in the debug-mode flowchart
    inputs: dict          # short summary shown on the node/detail panel
    outputs: dict          # short summary shown on the node/detail panel
    timing_ms: float
    detail: list | dict | None = None  # full drill-down data, shown on expand only
    # time.perf_counter() reading taken when this step began. Only differences
    # between steps mean anything (the UI subtracts the earliest one), and it's
    # what lets the timeline show whether steps ran one after another or
    # overlapped -- a step's own timing_ms can't tell you that.
    started_at: float | None = None


def summarize_candidates(items, score_attr: str, limit: int = 8) -> list[dict]:
    # `items` are retrieval/hybrid_search.py Candidate or retrieval/rrf.py
    # FusedResult objects -- both carry doc_id/heading and one scoring
    # attribute whose name/meaning differs by arm (see score_attr). Only
    # FusedResult carries `arms` (which search arm(s) found this chunk) --
    # included whenever present, since it's the direct answer to "was this
    # found by both arms" that otherwise has to be reverse-engineered from
    # the fused_score alone (see retrieval/rrf.py's 1/(60+rank) formula).
    summaries = []
    for c in items[:limit]:
        row = {
            "doc_id": c.doc_id,
            "heading": c.heading,
            score_attr: round(getattr(c, score_attr) or 0.0, 4),
        }
        arms = getattr(c, "arms", None)
        if arms is not None:
            row["arms"] = arms
        summaries.append(row)
    return summaries
