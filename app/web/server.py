"""FastAPI backend for the plain HTML/CSS/JS frontend in app/web/static/.

Replaces app/streamlit_app.py as the deployed UI (kept in the repo, unused,
as a fallback/reference -- see app/serve.py's docstring). Talks to chat.py
exactly the same way the Streamlit app did: this is a thin HTTP/SSE shell
around the same prepare_answer()/PendingAnswer.stream() pipeline, not a
reimplementation of any retrieval or generation logic.

Why Server-Sent Events instead of plain JSON: the UI streams the answer
token-by-token the same way the Streamlit app did with st.write_stream(), and
SSE is the simplest one-way "push text as it's generated" transport that
works over plain HTTP (no WebSocket upgrade, no extra library, and it
survives Caddy's reverse_proxy with zero special config).
"""
import dataclasses
import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat import ChatResponse, prepare_answer
from retrieval.query_understanding import FALLBACK_WARNING

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
LOG_PATH = os.path.join(LOGS_DIR, "requests.jsonl")
# Real-usage signal (docs/eval-and-feedback.md): a 👍/👎 (+ optional comment) on
# any past answer, keyed by that answer's request_id -- the same id already in
# requests.jsonl, so eval/triage_feedback.py can join the two on read without
# either file needing to know about the other's schema. A separate file, not a
# rewrite of the request's own log line, because a rating can arrive well after
# that line was written (the user reads the answer, then decides) and multiple
# ratings/comments for the same request_id are all kept, not overwritten --
# more signal, never destructive.
FEEDBACK_LOG_PATH = os.path.join(LOGS_DIR, "feedback.jsonl")
MAX_FEEDBACK_COMMENT_CHARS = 500

# Single pipeline, same choice app/streamlit_app.py made after Compare mode's
# headline turned out to swing on embedding-call network noise rather than
# anything the UI could act on -- see retrieval/pipeline.py's module
# docstring and INTERVIEW-QA.md for the measured before/after. "sequential"
# (Baseline) stays reachable from eval/run_eval.py and benchmarks only.
PIPELINE_MODE = "parallel"

app = FastAPI()


class HistoryTurn(BaseModel):
    query: str
    answer: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: str  # "up" | "down"
    comment: str | None = None


class ChatRequest(BaseModel):
    query: str
    role: str
    partner: str
    # Prior turns from this browser session, same role/partner only (the
    # frontend filters before sending -- see app.js's runQuery). Used so a
    # follow-up like "as shown in Dashboard idea#1" can be resolved at all;
    # see chat.py's module docstring for how retrieval and generation each
    # use it differently.
    history: list[HistoryTurn] = []


def _log_request(req: ChatRequest, result: ChatResponse, wall_ms: float, request_id: str) -> None:
    # Same lightweight observability app/streamlit_app.py's log_request()
    # wrote: one JSON line per request, excluding answer text/citations/
    # scores. Stands in for the OpenTelemetry->Langfuse tracing a production
    # system would use. The query-understanding fields below are counts and
    # categories only -- no additional user text beyond the `query` this log
    # already recorded before that stage existed.
    u = result.understanding or {}
    terms = u.get("corrected_terms", [])
    log_line = {
        "request_id": request_id,  # correlation id: also sent to the browser in the first SSE event
        "query": req.query,
        "role": req.role,
        "partner": req.partner,
        "abstained": result.abstained,
        "permission_refused": result.permission_refused,
        "degraded_rerank": result.degraded_rerank,
        "intent": u.get("intent"),
        "api_role": u.get("api_role"),
        "ambiguity": u.get("ambiguity"),
        "clarification": result.clarification,
        "normalization_count": len(terms),
        "corrected_terms_count": sum(1 for t in terms if t.get("reason") == "typo"),
        "understanding_fallback": FALLBACK_WARNING in u.get("warnings", []),
        "classifier": u.get("classifier"),  # "rules" | "llm" | "none"
        "llm_classifier_called": "llm_classification_ms" in result.timings_ms,
        "llm_classifier_fallback": any(w.startswith("llm_classifier_") for w in u.get("warnings", [])),  # called, but the rules' result was kept
        "retrieval_query_count": 1,  # always one search query (multi-query retrieval was tried and removed; see docs/query-understanding.md)
        "retrieval_result_count": result.retrieval_result_count,
        "retrieval_fallback_used": False,  # always False; kept so the log schema stays stable
        "llm_rewrite_attempted": result.llm_rewrite_attempted,  # the optional retry-on-abstain rewrite (QUERY_LLM_REWRITE_ENABLED) was tried
        "llm_rewrite_used": result.llm_rewrite_used,            # ...and its retry cleared the confidence gate
        "timings_ms": result.timings_ms,  # includes normalization_ms, classification_ms, llm_classification_ms, llm_rewrite_ms (each only when that step ran), understand_ms, retrieve/rerank, generate, total
        "total_wall_ms": wall_ms,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(log_line) + "\n")


def _sse(event: str, data: dict) -> str:
    # json.dumps escapes any real newlines inside `data` (e.g. mid-stream
    # answer text) into literal \n, so this is always exactly one line --
    # required by the SSE wire format (a bare newline would end the field).
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_chat(req: ChatRequest):
    # A plain (sync) generator, not async -- every call it makes (psycopg,
    # sentence-transformers, the OpenAI SDK) is blocking. FastAPI/Starlette
    # run a sync generator passed to StreamingResponse in a worker thread
    # automatically, so this never blocks the event loop despite not using
    # async/await anywhere in the pipeline.
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    yield _sse("status", {"phase": "retrieving", "request_id": request_id})

    history = [{"query": h.query, "answer": h.answer} for h in req.history]
    prepared = prepare_answer(req.query, req.role, req.partner, mode=PIPELINE_MODE, history=history)

    if isinstance(prepared, ChatResponse):
        # Permission refusal or confidence-gate abstain -- nothing to stream.
        _log_request(req, prepared, round((time.perf_counter() - t0) * 1000, 1), request_id)
        yield _sse("final", {**dataclasses.asdict(prepared), "request_id": request_id})
        return

    retrieve_s = time.perf_counter() - t0
    yield _sse("retrieved", {"chunks": len(prepared.result.top_k), "elapsed_s": round(retrieve_s, 1)})

    for delta in prepared.stream():
        yield _sse("delta", {"text": delta})

    result = prepared.finish()
    _log_request(req, result, result.timings_ms["total_ms"], request_id)
    yield _sse("final", {**dataclasses.asdict(result), "request_id": request_id})


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(_stream_chat(req), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    # request_id is generated server-side per request (see _stream_chat) and
    # is exactly a uuid4 hex slice -- reject anything else rather than write
    # arbitrary client-supplied text into the log's join key.
    if req.rating not in ("up", "down"):
        raise HTTPException(422, "rating must be 'up' or 'down'")
    # request_id is always uuid.uuid4().hex[:12] (see _stream_chat) -- exact length, lowercase hex only.
    if len(req.request_id) != 12 or not all(c in "0123456789abcdef" for c in req.request_id):
        raise HTTPException(422, "invalid request_id")
    comment = (req.comment or "").strip()[:MAX_FEEDBACK_COMMENT_CHARS] or None
    line = {"request_id": req.request_id, "rating": req.rating, "comment": comment, "ts": time.time()}
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return {"ok": True}


# Static files (index.html, style.css, app.js) mounted last -- FastAPI/
# Starlette match routes in registration order, so /api/chat above always
# wins for that exact path; everything else falls through to this mount.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
