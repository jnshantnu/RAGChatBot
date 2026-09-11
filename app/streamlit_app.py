"""Streamlit chat UI: calls chat.answer_query() directly, in-process -- no
FastAPI layer, no HTTP hop. Same lightweight request logging app/main.py
used to do, moved here since this is now the only entry point."""
import json
import os
import sys
import time

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat import answer_query

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "requests.jsonl")

st.set_page_config(page_title="Partner AI Chat Bot", page_icon="\U0001F4C4", layout="wide")


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
st.info(
    "**Demo only.** Built on publicly available Autodesk Partner WebServices "
    "reference documentation. Not an official Autodesk product; not affiliated "
    "with or endorsed by Autodesk.",
    icon="ℹ️",
)

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
    st.divider()
    debug_mode = st.checkbox(
        "Debug mode", value=False,
        help="Show the retrieval/generation pipeline as a flowchart, with real inputs/outputs for the query just run.",
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


# Fixed pipeline shape -- which node feeds which, and the step number each
# carries in the diagram -- independent of any one query's data. Edges/numbers
# for a step that didn't run this time (e.g. "Generate Answer" on an abstain)
# are simply skipped below.
_FLOW_EDGES = [
    ("Query Rewrite", "Embed Query"), ("Query Rewrite", "Keyword Search"),
    ("Embed Query", "Semantic Search"),
    ("Keyword Search", "RRF Fusion"), ("Semantic Search", "RRF Fusion"),
    ("RRF Fusion", "Rerank (competitive)"),
    ("Guaranteed Fetch", "Rerank (guaranteed)"),
    ("Rerank (competitive)", "Confidence Gate"), ("Rerank (guaranteed)", "Confidence Gate"),
    ("Confidence Gate", "Generate Answer"),
]
_STEP_ORDER = [
    "Query Rewrite", "Embed Query", "Keyword Search", "Semantic Search", "RRF Fusion",
    "Rerank (competitive)", "Guaranteed Fetch", "Rerank (guaranteed)", "Confidence Gate", "Generate Answer",
]
_NODE_COLORS = {
    "Query Rewrite": "#1c2b42", "Embed Query": "#c15d52",
    "Keyword Search": "#1c2b42", "Semantic Search": "#1c2b42",
    "RRF Fusion": "#5c6b80", "Rerank (competitive)": "#c15d52",
    "Guaranteed Fetch": "#2f6690", "Rerank (guaranteed)": "#2f6690",
    "Confidence Gate": "#b0413a", "Generate Answer": "#c15d52",
}


# The flowchart node is a glanceable summary, not a data dump -- the actual
# values (full rewritten query text, full citation list, etc.) live in the
# "Full detail" card for this step below the diagram. A node only ever shows:
# a list's length (never its contents), a short scalar's actual value (it's
# already compact enough to be useful at a glance), or just the field name
# with no value at all when that value is a long string.
_MAX_INLINE_VALUE_LEN = 18


def _node_label_line(key, value):
    if isinstance(value, list):
        return f"{key}: {len(value)}"
    s = str(value)
    if len(s) > _MAX_INLINE_VALUE_LEN:
        return key  # value too long for the node itself -- see the detail card
    return f"{key}: {s}"


# Extra subtitle line shown only on nodes named here -- lets a node's label
# match the terminology used in the case-study slide deck it corresponds to
# (only "Rerank (competitive)" has a slide equivalent; "Rerank (guaranteed)"
# doesn't, per that deck's own "no case-study equivalent" annotation).
_NODE_SUBTITLES = {"Rerank (competitive)": "Cross-Encoder Rerank"}


# Fixed pixel layout (center x, center y) -- hand-placed, not auto-computed,
# since the topology never changes: only the data inside each box does.
# Three horizontal lanes (top/mid/bottom) so the two parallel sub-pipelines
# (competitive: embed->search->fuse->rerank; guaranteed: fetch->rerank) are
# visually distinct before they both merge into the confidence gate.
# Lanes are 170px apart center-to-center -- box height is 130px (half-height
# 65), so that's a real 40px gap between edges, not just a round number.
# (The previous version used 120px lane spacing against a 130px box height --
# a 10px vertical overlap by construction, which is what caused boxes to sit
# on top of each other.)
_NODE_POS = {
    "Query Rewrite": (130, 270),
    "Embed Query": (420, 100), "Keyword Search": (420, 270), "Guaranteed Fetch": (420, 440),
    "Semantic Search": (710, 100), "Rerank (guaranteed)": (710, 440),
    "RRF Fusion": (1000, 185),
    "Rerank (competitive)": (1290, 185),
    "Confidence Gate": (1580, 270),
    "Generate Answer": (1870, 270),
}
_BOX_W, _BOX_H = 240, 130


def _wrap(text, max_chars):
    words = text.split()
    lines, current = [], ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_flow_svg(trace):
    steps_by_name = {s.name: s for s in trace}
    numbers = {name: i + 1 for i, name in enumerate(_STEP_ORDER)}
    max_x = max(x for name, (x, _) in _NODE_POS.items() if name in steps_by_name) + _BOX_W // 2 + 40
    max_y = max(y for name, (_, y) in _NODE_POS.items() if name in steps_by_name) + _BOX_H // 2 + 40

    svg = [f'<svg viewBox="0 0 {max_x} {max_y}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">']
    svg.append("""<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#9aa5b1"/></marker></defs>""")

    # Edges first, so nodes draw on top of the lines.
    for a, b in _FLOW_EDGES:
        if a in steps_by_name and b in steps_by_name:
            ax, ay = _NODE_POS[a]
            bx, by = _NODE_POS[b]
            x1, y1 = ax + _BOX_W // 2, ay
            x2, y2 = bx - _BOX_W // 2, by
            svg.append(f'<path d="M{x1},{y1} C{(x1+x2)//2},{y1} {(x1+x2)//2},{y2} {x2},{y2}" '
                       f'fill="none" stroke="#9aa5b1" stroke-width="2.5" marker-end="url(#arrow)"/>')

    for name, step in steps_by_name.items():
        cx, cy = _NODE_POS[name]
        color = _NODE_COLORS.get(name, "#1c2b42")
        title_lines = _wrap(f"{numbers[name]}. {name.upper()}", 22)
        body_lines = []
        if name in _NODE_SUBTITLES:
            body_lines.append(_NODE_SUBTITLES[name])
        for k, v in step.outputs.items():
            body_lines.append(_node_label_line(k, v))
        body_lines.append(f"⏱ {step.timing_ms} ms")

        svg.append(f'<g class="trace-node" tabindex="0">')
        svg.append(f'<rect x="{cx - _BOX_W//2}" y="{cy - _BOX_H//2}" width="{_BOX_W}" height="{_BOX_H}" '
                   f'rx="14" fill="{color}"/>')
        ty = cy - _BOX_H // 2 + 16 + 13 * (len(title_lines) - 1) / 2
        text_x = cx - _BOX_W // 2 + 16
        svg.append(f'<text x="{text_x}" y="{ty - 13 * (len(title_lines) - 1) / 2}" fill="white" font-size="14" font-weight="700">')
        for i, line in enumerate(title_lines):
            svg.append(f'<tspan x="{text_x}" dy="{0 if i == 0 else 17}">{line}</tspan>')
        svg.append('</text>')
        body_start = ty + 17 * len(title_lines) - (13 * (len(title_lines) - 1) / 2) + 6
        svg.append(f'<text x="{text_x}" y="{body_start}" fill="#e8ecf1" font-size="12">')
        for i, line in enumerate(body_lines):
            svg.append(f'<tspan x="{text_x}" dy="{0 if i == 0 else 18}">{line}</tspan>')
        svg.append('</text>')
        svg.append('</g>')

    svg.append("</svg>")
    svg_body = "\n".join(svg)

    return f"""
    <style>
      body {{ margin: 0; }}
      .flow-wrap {{ overflow-x: auto; overflow-y: hidden; padding: 4px; }}
      .trace-node {{
        transform-box: fill-box;
        transform-origin: center;
        transition: transform 0.18s ease;
        cursor: pointer;
      }}
      .trace-node:hover {{
        transform: scale(1.45);
      }}
    </style>
    <div class="flow-wrap">{svg_body}</div>
    """


_DETAIL_ROW_SIZE = 5  # steps per row of side-by-side cards, instead of one long vertical stack


def render_debug_trace(trace):
    if not trace:
        return
    components.html(build_flow_svg(trace), height=610, scrolling=True)
    numbers = {name: i + 1 for i, name in enumerate(_STEP_ORDER)}
    for row_start in range(0, len(trace), _DETAIL_ROW_SIZE):
        row = trace[row_start : row_start + _DETAIL_ROW_SIZE]
        columns = st.columns(len(row), gap="small")
        for column, step in zip(columns, row):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{numbers[step.name]}. {step.name}**")
                    st.caption(f"⏱ {step.timing_ms} ms")
                    with st.expander("Inputs"):
                        st.json(step.inputs, expanded=False)
                    with st.expander("Outputs"):
                        st.json(step.outputs, expanded=False)
                    if step.detail is not None:
                        with st.expander("Full detail"):
                            st.json(step.detail, expanded=False)


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

    if debug_mode and result.trace:
        st.divider()
        st.caption("Debug mode — pipeline flowchart for this query")
        render_debug_trace(result.trace)


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
