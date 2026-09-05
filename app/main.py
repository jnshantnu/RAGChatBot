import json
import os
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat import answer_query

app = FastAPI(title="RAG Chatbot POC")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "requests.jsonl")


class ChatRequest(BaseModel):
    query: str
    role: str = "reseller"
    partner: str = "acme"


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    result = answer_query(req.query, req.role, req.partner)
    log_line = {
        "query": req.query,
        "role": req.role,
        "partner": req.partner,
        "abstained": result.abstained,
        "permission_refused": result.permission_refused,
        "degraded_rerank": result.degraded_rerank,
        "timings_ms": result.timings_ms,
        "total_wall_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(log_line) + "\n")
    return result


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
