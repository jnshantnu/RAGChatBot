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
from chat import ChatResponse, answer_query, prepare_answer
from retrieval.rerank import TUNED_MAX_LENGTH
from retrieval.trace import PIPELINE_STEP_ORDER

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "requests.jsonl")

st.set_page_config(page_title="Partner AI Chat Bot", page_icon="\U0001F4C4", layout="wide")


@st.cache_resource(show_spinner="Loading the reranker model (one-time per server start, ~30s)...")
def _warm_reranker():
    # retrieval/rerank.py already caches each loaded model at module level, so
    # this only pays the torch/sentence-transformers import + model-load cost
    # once per running process -- but calling it here, at page-load time,
    # means whoever visits first sees a spinner instead of it silently eating
    # 30s of their first query's rerank_ms. Only the tuned-length variant is
    # loaded: it's the one the app's pipeline uses (the default-length model
    # is only needed by eval/benchmarks, which load it on demand).
    from retrieval.rerank import TUNED_MAX_LENGTH, _get_model

    return _get_model(TUNED_MAX_LENGTH)


_warm_reranker()

# CSS injection on top of .streamlit/config.toml's base theme -- everything
# the theme API doesn't reach. Values (font, colors, radii, borders) are
# copied from /root/alerts-dashboard's own stylesheets (portfolio.css,
# sidebar.css, discord-chat.css), at the user's request to match that app's
# look, not guessed. Targets Streamlit's own `data-testid` hooks (confirmed
# against the real rendered DOM), which are more stable release to release
# than its auto-generated CSS class names, but are still an internal
# implementation detail, not a public API -- a future Streamlit upgrade can
# rename or drop one. If the UI ever looks partially unstyled after an
# upgrade, this block is the first place to check.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* Header: alerts-dashboard's frosted sticky bar (--bg-header, blur(12px)) */
[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.85); backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

/* Sidebar: alerts-dashboard's hub-sidebar rail. Section header styled like
   .hub-sidebar-section (10px uppercase, letter-spaced, 65% opacity) */
[data-testid="stSidebar"] { border-right: 1px solid rgba(0, 0, 0, 0.1); }
[data-testid="stSidebar"] h2 {
    font-size: 10px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
    color: #64748b; opacity: 0.65; margin-bottom: 4px;
}

/* Two-tier surface system, straight from discord-chat.css: pure-white cards
   for content (chat bubbles, bordered score/timing panels) vs. a slightly
   tinted "panel" tone (--dc-panel-bg: #eef1f6) for toolbar-like chrome (the
   retrieval status widget) -- same distinction their Messages/Digest toolbar
   draws against their message cards. */
[data-testid="stChatMessage"], [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 14px;
}
[data-testid="stChatMessage"] { padding: 4px 6px; margin-bottom: 10px; }
[data-testid="stStatusWidget"] {
    background: #eef1f6; border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 14px;
}

/* Chat avatars: rounded-square icon badges like .acct-icon (26px, 8px
   radius, bg var(--bg-glass), hairline border) rather than plain circles */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    background: rgba(0, 0, 0, 0.03) !important; border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 8px !important;
}

[data-testid="stAlertContainer"] { border-radius: 12px; }
[data-testid="stExpander"] { border-radius: 12px; border-color: rgba(0, 0, 0, 0.1); }

/* Title-row dot: alerts-dashboard's .logo-glow -- a small pulsing cyan dot
   next to the page name (same @keyframes pulse-cyan as portfolio.css) */
@keyframes pulse-cyan {
    0% { box-shadow: 0 0 0 0 rgba(0, 150, 199, 0.5); }
    70% { box-shadow: 0 0 0 8px rgba(0, 150, 199, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 150, 199, 0); }
}
.rag-title-dot {
    width: 10px; height: 10px; border-radius: 50%; background: #0096c7;
    box-shadow: 0 0 8px #0096c7; animation: pulse-cyan 2s infinite; display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
    '<span class="rag-title-dot"></span>'
    '<span style="font-size:26px; font-weight:600; letter-spacing:0.5px; color:#0f172a;">Partner AI Chat Bot</span>'
    '</div>',
    unsafe_allow_html=True,
)
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
    debug_mode = st.toggle(
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
_STEP_ORDER = PIPELINE_STEP_ORDER
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


# Fixed layout: (center x, lane). The topology never changes, only the data inside
# each box does, so positions are hand-placed rather than auto-computed. Three
# lanes (0/1/2) keep the two sub-pipelines (competitive: embed->search->fuse->
# rerank; guaranteed: fetch->rerank) visually distinct until they merge at the
# confidence gate. The pixel y of a lane is computed in build_flow_svg from the
# tallest box actually present, so boxes can never overlap however many output
# lines a step has (an earlier fixed-height layout let the taller Optimized
# rerank box spill out of its own rectangle).
#
# NOTE this diagram is a dependency map -- what feeds what -- and is identical
# for Baseline and Optimized on purpose. WHEN steps ran (one after another vs
# overlapping) is shown by each box's "starts +N ms" line and the timeline
# under the chart, not by where boxes sit.
_NODE_POS = {
    "Query Rewrite": (130, 1),
    "Embed Query": (420, 0), "Keyword Search": (420, 1), "Guaranteed Fetch": (420, 2),
    "Semantic Search": (710, 0), "Rerank (guaranteed)": (710, 2),
    "RRF Fusion": (1000, 0.5),
    "Rerank (competitive)": (1290, 0.5),
    "Confidence Gate": (1580, 1),
    "Generate Answer": (1870, 1),
}
_BOX_W = 240
_LANE_GAP = 40  # empty space between two boxes' edges in adjacent lanes
_BATCH_COLOR = "#f0a030"


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


def _trace_origin(trace):
    starts = [s.started_at for s in trace if s.started_at is not None]
    return min(starts) if starts else None


def _start_offset_ms(step, origin):
    if step.started_at is None or origin is None:
        return None
    return round((step.started_at - origin) * 1000)


def build_flow_svg(trace):
    steps_by_name = {s.name: s for s in trace}
    numbers = {name: i + 1 for i, name in enumerate(_STEP_ORDER)}
    origin = _trace_origin(trace)

    # Box content first, so every box can be sized to fit its own text.
    content = {}
    for name, step in steps_by_name.items():
        title_lines = _wrap(f"{numbers[name]}. {name.upper()}", 22)
        body_lines = []
        if name in _NODE_SUBTITLES:
            body_lines.append(_NODE_SUBTITLES[name])
        for k, v in step.outputs.items():
            body_lines.append(_node_label_line(k, v))
        start = _start_offset_ms(step, origin)
        body_lines.append(f"⏱ {step.timing_ms} ms" + (f" · starts +{start} ms" if start is not None else ""))
        content[name] = (title_lines, body_lines)

    def needed_height(title_lines, body_lines):
        return 16 + 17 * len(title_lines) + 8 + 18 * len(body_lines) + 8

    box_h = max(needed_height(*c) for c in content.values())
    lane_pitch = box_h + _LANE_GAP
    pad = 20

    def center_y(name):
        return pad + box_h / 2 + _NODE_POS[name][1] * lane_pitch

    max_x = max(_NODE_POS[n][0] for n in steps_by_name) + _BOX_W // 2 + 40
    max_y = max(center_y(n) for n in steps_by_name) + box_h / 2 + pad

    svg = [f'<svg viewBox="0 0 {max_x} {max_y:.0f}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">']
    svg.append("""<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#9aa5b1"/></marker></defs>""")

    # Edges first, so nodes draw on top of the lines.
    for a, b in _FLOW_EDGES:
        if a in steps_by_name and b in steps_by_name:
            x1, y1 = _NODE_POS[a][0] + _BOX_W // 2, center_y(a)
            x2, y2 = _NODE_POS[b][0] - _BOX_W // 2, center_y(b)
            svg.append(f'<path d="M{x1},{y1:.0f} C{(x1+x2)//2},{y1:.0f} {(x1+x2)//2},{y2:.0f} {x2},{y2:.0f}" '
                       f'fill="none" stroke="#9aa5b1" stroke-width="2.5" marker-end="url(#arrow)"/>')

    for name, step in steps_by_name.items():
        title_lines, body_lines = content[name]
        cx, cy = _NODE_POS[name][0], center_y(name)
        color = _NODE_COLORS.get(name, "#1c2b42")
        top = cy - box_h / 2
        left = cx - _BOX_W // 2
        # Both rerank nodes of an Optimized run are ONE model call; the shared
        # orange dashed outline is how the chart says so.
        outline = (f' stroke="{_BATCH_COLOR}" stroke-width="4" stroke-dasharray="9 5"'
                   if step.outputs.get("batched_call") else "")

        svg.append('<g class="trace-node" tabindex="0">')
        svg.append(f'<rect x="{left}" y="{top:.0f}" width="{_BOX_W}" height="{box_h:.0f}" rx="14" fill="{color}"{outline}/>')
        y = top + 26
        svg.append(f'<text x="{left + 16}" y="{y:.0f}" fill="white" font-size="14" font-weight="700">')
        for i, line in enumerate(title_lines):
            svg.append(f'<tspan x="{left + 16}" dy="{0 if i == 0 else 17}">{line}</tspan>')
        svg.append('</text>')
        y += 17 * (len(title_lines) - 1) + 24
        svg.append(f'<text x="{left + 16}" y="{y:.0f}" fill="#e8ecf1" font-size="12">')
        for i, line in enumerate(body_lines):
            svg.append(f'<tspan x="{left + 16}" dy="{0 if i == 0 else 18}">{line}</tspan>')
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


def _timeline_rows(trace):
    origin = _trace_origin(trace)
    if origin is None:
        return [], 0
    rows = []
    for name in _STEP_ORDER:
        step = next((s for s in trace if s.name == name), None)
        if step is None or step.started_at is None:
            continue
        start = (step.started_at - origin) * 1000
        rows.append((name, start, step.timing_ms, bool(step.outputs.get("batched_call"))))
    total = max((start + dur for _, start, dur, _ in rows), default=0)
    return rows, total


def _nice_tick(axis_max_ms):
    for tick in (100, 200, 500, 1000, 2000, 5000, 10000):
        if axis_max_ms / tick <= 8:
            return tick
    return 20000


def build_timeline_html(sections):
    """Waterfall of when each step actually ran. `sections` is a list of
    (title, trace); sections share one time axis (the longest run), so bar
    lengths stay comparable if more than one is passed."""
    built = [(title, *_timeline_rows(trace)) for title, trace in sections]
    axis_max = max((total for _, _, total in built), default=0) * 1.03 or 1
    tick = _nice_tick(axis_max)
    ticks = [t for t in range(0, int(axis_max) + 1, tick)]

    def grid():
        return "".join(f'<i style="left:{t / axis_max * 100:.2f}%"></i>' for t in ticks)

    html = ["""<style>
      body { margin: 0; font-family: Helvetica, Arial, sans-serif; }
      .tl { background: #f7f8fa; color: #1c2b42; border-radius: 12px; padding: 14px 18px 10px; margin-bottom: 12px; }
      .tl h4 { margin: 0 0 8px; font-size: 15px; }
      .tl h4 span { font-weight: 400; color: #5c6b80; font-size: 13px; }
      .row { display: grid; grid-template-columns: 190px 1fr 80px; align-items: center; height: 24px; font-size: 13px; }
      .row .val { text-align: right; color: #5c6b80; font-variant-numeric: tabular-nums; }
      .track { position: relative; height: 16px; }
      .track i { position: absolute; top: 0; bottom: 0; border-left: 1px solid #dde2e8; }
      .bar { position: absolute; top: 2px; height: 12px; border-radius: 3px; min-width: 3px; }
      .axis { display: grid; grid-template-columns: 190px 1fr 80px; font-size: 11px; color: #5c6b80; height: 16px; }
      .axis .scale { position: relative; }
      .axis .scale b { position: absolute; font-weight: 400; transform: translateX(-50%); }
      .legend { font-size: 12px; color: #5c6b80; margin-top: 4px; }
    </style>"""]
    numbers = {name: i + 1 for i, name in enumerate(_STEP_ORDER)}
    for title, rows, total in built:
        html.append(f'<div class="tl"><h4>{title} <span>— ended at {total / 1000:.1f} s</span></h4>')
        labels = "".join(f'<b style="left:{t / axis_max * 100:.2f}%">{t / 1000:g}s</b>' for t in ticks)
        html.append(f'<div class="axis"><span></span><div class="scale">{labels}</div><span></span></div>')
        for name, start, dur, batched in rows:
            color = _NODE_COLORS.get(name, "#1c2b42")
            border = f"outline:2px dashed {_BATCH_COLOR};" if batched else ""
            html.append(
                f'<div class="row"><span>{numbers[name]}. {name}</span>'
                f'<div class="track">{grid()}<div class="bar" style="left:{start / axis_max * 100:.2f}%;'
                f'width:{dur / axis_max * 100:.2f}%;background:{color};{border}"></div></div>'
                f'<span class="val">{dur:,.0f} ms</span></div>'
            )
        html.append("</div>")
    html.append(
        f'<div class="legend"><span style="color:{_BATCH_COLOR}">▭ dashed orange</span> = these steps were one shared '
        "model call. A bar's left edge is when the step started; steps whose bars overlap ran at the same time.</div>"
    )
    # iframe height: per section ~80px of title/axis/padding plus 24px per row, plus the legend.
    return "".join(html), 40 + sum(80 + 24 * len(rows) for _, rows, _ in built)


_PIPELINE_BLURB = (
    "Embed Query, Keyword Search and Guaranteed Fetch start together (see each box's “starts +N ms”). "
    "The two rerank steps are one batched model call (dashed orange), and the reranker reads up to "
    f"{TUNED_MAX_LENGTH} tokens per chunk."
)


def render_timeline(sections):
    html, height = build_timeline_html(sections)
    components.html(html, height=height, scrolling=True)


_DETAIL_ROW_SIZE = 5  # steps per row of side-by-side cards, instead of one long vertical stack


def render_debug_trace(trace):
    if not trace:
        return
    st.caption(_PIPELINE_BLURB)
    st.caption("The boxes show what feeds what; the timeline below shows when each step actually ran.")
    components.html(build_flow_svg(trace), height=610, scrolling=True)
    render_timeline([("Timeline", trace)])
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


def _tags_markdown(abstained=False, refused=False, degraded=False):
    tags = []
    if abstained:
        tags.append(":red-background[abstained]")
    if refused:
        tags.append(":orange-background[permission refused]")
    if degraded:
        tags.append(":gray-background[degraded.rerank]")
    return " ".join(tags)


def render_result(result):
    tags = _tags_markdown(result.abstained, result.permission_refused, result.degraded_rerank)
    if tags:
        st.markdown(tags)

    st.markdown(result.answer)
    render_result_details(result)


def render_result_details(result):
    # Everything below the answer text -- split out so the streaming path,
    # which has already written the answer by the time a ChatResponse exists,
    # can render just this part.
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


PIPELINE_MODE = "parallel"  # retrieval/pipeline.py's "Optimized" strategy; the sequential Baseline is kept for eval and benchmarks only


def _stream_single(query, role, partner):
    # Retrieval runs first (behind a spinner); if the answer is warranted, the
    # LLM's text streams into the bubble as it's written instead of appearing
    # all at once after the full generation. Refusals and abstains have nothing
    # to stream and render immediately.
    t0 = time.perf_counter()
    with st.spinner("Retrieving..."):
        prepared = prepare_answer(query, role, partner, mode=PIPELINE_MODE)

    if isinstance(prepared, ChatResponse):
        result = prepared
        log_request(query, role, partner, result, round((time.perf_counter() - t0) * 1000, 1))
        render_result(result)
    else:
        tags = _tags_markdown(degraded=prepared.result.degraded_rerank)
        if tags:
            st.markdown(tags)
        st.write_stream(prepared.stream())
        result = prepared.finish()
        log_request(query, role, partner, result, result.timings_ms["total_ms"])
        first = result.timings_ms.get("first_token_ms")
        if first is not None:
            st.caption(
                f"First words after {first / 1000:.1f}s · full answer in {result.timings_ms['total_ms'] / 1000:.1f}s"
            )
        render_result_details(result)

    return {"query": query, "role": role, "partner": partner, "result": result}


def render_turn(turn):
    render_result(turn["result"])


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{turn['role']} @ {turn['partner']}:** {turn['query']}")
    with st.chat_message("assistant"):
        render_turn(turn)

query = st.chat_input("Ask a question, e.g. 'How do I authenticate to the Partner WebServices API?'")
if query:
    with st.chat_message("user"):
        st.markdown(f"**{role} @ {partner}:** {query}")

    with st.chat_message("assistant"):
        turn = _stream_single(query, role, partner)

    st.session_state.history.append(turn)
