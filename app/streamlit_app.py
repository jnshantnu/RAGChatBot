"""Streamlit chat UI: calls chat.answer_query() directly, in-process -- no
FastAPI layer, no HTTP hop. Same lightweight request logging app/main.py
used to do, moved here since this is now the only entry point."""
import json
import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat import answer_query

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "requests.jsonl")

st.set_page_config(page_title="Partner AI Chat Bot", page_icon="\U0001F4C4", layout="centered")


@st.cache_resource(show_spinner="Loading the reranker model (one-time per server start, ~30s)...")
def _warm_reranker():
    # retrieval/rerank.py already caches the loaded model at module level, so
    # this only pays the torch/sentence-transformers import + model-load cost
    # once per running process -- but calling it here, at page-load time,
    # means whoever visits first sees a spinner instead of it silently eating
    # 30s of their first query's rerank_ms.
    from retrieval.rerank import _get_model

    return _get_model()


_warm_reranker()

st.title("Partner AI Chat Bot")

with st.sidebar:
    st.header("Session")
    partner = st.selectbox(
        "Partner", ["acme", "globex"],
        format_func=lambda p: {"acme": "Partner A", "globex": "Partner B"}[p],
    )
    role = st.selectbox(
        "Role", ["reseller", "distributor", "solution_provider", "principal"],
        format_func=lambda r: {
            "reseller": "Reseller", "distributor": "Distributor",
            "solution_provider": "Solution Provider", "principal": "Principal",
        }[r],
    )

if "history" not in st.session_state:
    st.session_state.history = []


def log_request(query, role, partner, result, wall_ms):
    # Same lightweight observability as the old /chat endpoint: one JSON line
    # per request, excluding answer text/citations/scores. Stands in for the
    # OpenTelemetry->Langfuse tracing a production system would use.
    log_line = {
        "query": query,
        "role": role,
        "partner": partner,
        "abstained": result.abstained,
        "permission_refused": result.permission_refused,
        "degraded_rerank": result.degraded_rerank,
        "timings_ms": result.timings_ms,
        "total_wall_ms": wall_ms,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(log_line) + "\n")


def render_result(result):
    tags = []
    if result.abstained:
        tags.append(":red-background[abstained]")
    if result.permission_refused:
        tags.append(":orange-background[permission refused]")
    if result.degraded_rerank:
        tags.append(":gray-background[degraded.rerank]")
    if tags:
        st.markdown(" ".join(tags))

    st.markdown(result.answer)

    if result.citations:
        st.caption("Citations: " + ", ".join(result.citations))

    if result.rewritten_query:
        with st.expander(f"Rewritten query ({', '.join(result.rewrite_rules_applied)})"):
            st.write(result.rewritten_query)

    if result.scores:
        with st.expander(f"Retrieval scores ({len(result.scores)})"):
            st.dataframe(
                [
                    {
                        "doc": s["doc_id"],
                        "heading": s["heading"],
                        "rerank score": round(s["rerank_score"], 4) if s.get("rerank_score") is not None else "-",
                        "fused score": round(s["fused_score"], 4),
                        "arms": ", ".join(s["arms"]),
                    }
                    for s in result.scores
                ],
                use_container_width=True,
                hide_index=True,
            )

    if result.timings_ms:
        with st.expander("Timings"):
            st.json(result.timings_ms)


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{turn['role']} @ {turn['partner']}:** {turn['query']}")
    with st.chat_message("assistant"):
        render_result(turn["result"])

query = st.chat_input("Ask a question, e.g. 'How do I authenticate to the Partner WebServices API?'")
if query:
    with st.chat_message("user"):
        st.markdown(f"**{role} @ {partner}:** {query}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.perf_counter()
            result = answer_query(query, role, partner)
            wall_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_request(query, role, partner, result, wall_ms)
        render_result(result)

    st.session_state.history.append({"query": query, "role": role, "partner": partner, "result": result})
