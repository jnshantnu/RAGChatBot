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

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat import ChatResponse, prepare_answer

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "requests.jsonl")

# Single pipeline, same choice app/streamlit_app.py made after Compare mode's
# headline turned out to swing on embedding-call network noise rather than
# anything the UI could act on -- see retrieval/pipeline.py's module
# docstring and INTERVIEW-QA.md for the measured before/after. "sequential"
# (Baseline) stays reachable from eval/run_eval.py and benchmarks only.
PIPELINE_MODE = "parallel"

app = FastAPI()


class ChatRequest(BaseModel):
    query: str
    role: str
    partner: str


def _log_request(req: ChatRequest, result: ChatResponse, wall_ms: float) -> None:
    # Same lightweight observability app/streamlit_app.py's log_request()
    # wrote: one JSON line per request, excluding answer text/citations/
    # scores. Stands in for the OpenTelemetry->Langfuse tracing a production
    # system would use.
    log_line = {
        "query": req.query,
        "role": req.role,
        "partner": req.partner,
        "abstained": result.abstained,
        "permission_refused": result.permission_refused,
        "degraded_rerank": result.degraded_rerank,
        "timings_ms": result.timings_ms,
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
    yield _sse("status", {"phase": "retrieving"})

    prepared = prepare_answer(req.query, req.role, req.partner, mode=PIPELINE_MODE)

    if isinstance(prepared, ChatResponse):
        # Permission refusal or confidence-gate abstain -- nothing to stream.
        _log_request(req, prepared, round((time.perf_counter() - t0) * 1000, 1))
        yield _sse("final", dataclasses.asdict(prepared))
        return

    retrieve_s = time.perf_counter() - t0
    yield _sse("retrieved", {"chunks": len(prepared.result.top_k), "elapsed_s": round(retrieve_s, 1)})

    for delta in prepared.stream():
        yield _sse("delta", {"text": delta})

    result = prepared.finish()
    _log_request(req, result, result.timings_ms["total_ms"])
    yield _sse("final", dataclasses.asdict(result))


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(_stream_chat(req), media_type="text/event-stream")


# Static files (index.html, style.css, app.js) mounted last -- FastAPI/
# Starlette match routes in registration order, so /api/chat above always
# wins for that exact path; everything else falls through to this mount.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
