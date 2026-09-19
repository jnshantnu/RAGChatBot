// Plain vanilla JS, no build step and no framework -- deploy is "copy files,
// restart the server," matching app/serve.py's existing philosophy. Talks to
// /api/chat (app/web/server.py) via a hand-rolled SSE reader (EventSource
// can't send a POST body, which is why this isn't just `new EventSource(...)`).
//
// Security note: every piece of text that could contain user input or LLM
// output (query text, rewritten query, citations, doc headings, chunk text,
// the answer itself) is inserted via DOM textContent, or through esc()
// before being placed into an HTML/SVG template string -- never raw string
// interpolation into innerHTML. The debug flowchart/timeline are built as
// big template strings for simplicity (matching how app/streamlit_app.py's
// Python version built them), so esc() is what keeps that safe.

const PIPELINE_STEP_ORDER = [
  "Query Rewrite", "Embed Query", "Keyword Search", "Semantic Search", "RRF Fusion",
  "Rerank (competitive)", "Guaranteed Fetch", "Rerank (guaranteed)", "Confidence Gate", "Generate Answer",
];

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

const logEl = document.getElementById("log");
const composer = document.getElementById("composer");
const queryInput = document.getElementById("query");
const partnerSelect = document.getElementById("partner");
const roleSelect = document.getElementById("role");
const debugToggle = document.getElementById("debug");
const sidebarToggle = document.getElementById("sidebarToggle");

const history = []; // finished turns: {query, role, partner, response}

debugToggle.addEventListener("change", renderHistory);

// Narrow-screen sidebar: an off-canvas overlay toggled by the hamburger
// button (only visible below 900px, see style.css), closed by tapping
// its own backdrop or pressing Escape -- desktop never sees this class.
sidebarToggle.addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
document.addEventListener("click", (e) => {
  if (!document.body.classList.contains("sidebar-open")) return;
  if (e.target.closest(".sidebar") || e.target === sidebarToggle) return;
  document.body.classList.remove("sidebar-open");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.body.classList.remove("sidebar-open");
});

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  runQuery(query, roleSelect.value, partnerSelect.value);
});

async function runQuery(query, role, partner) {
  composer.querySelector("button").disabled = true;

  // Live turn: user bubble + an assistant card updated in place as SSE
  // events arrive. Once the 'final' event lands, this DOM is discarded and
  // replaced by a full renderHistory() pass, so streaming and replay share
  // exactly one rendering code path for the finished result.
  const userTurn = document.createElement("div");
  userTurn.className = "turn user";
  userTurn.innerHTML = `<div class="avatar">🧑</div><div class="card"></div>`;
  const userCard = userTurn.querySelector(".card");
  const b = document.createElement("b");
  b.textContent = `${role} @ ${partner}: `;
  userCard.appendChild(b);
  userCard.appendChild(document.createTextNode(query));
  logEl.appendChild(userTurn);

  const liveTurn = document.createElement("div");
  liveTurn.className = "turn assistant";
  liveTurn.innerHTML = `<div class="avatar">🤖</div><div class="card"></div>`;
  const liveCard = liveTurn.querySelector(".card");
  const statusLine = document.createElement("div");
  statusLine.className = "status-line";
  statusLine.innerHTML = `<span class="spinner"></span><span>Retrieving relevant context...</span>`;
  liveCard.appendChild(statusLine);
  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  liveCard.appendChild(answerEl);
  logEl.appendChild(liveTurn);
  liveTurn.scrollIntoView({ behavior: "smooth", block: "end" });

  const t0 = performance.now();
  let answerText = "";

  try {
    const resp = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, role, partner }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const evLine = block.split("\n").find((l) => l.startsWith("event: "));
        const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
        if (!evLine || !dataLine) continue;
        const event = evLine.slice(7);
        const data = JSON.parse(dataLine.slice(6));

        if (event === "retrieved") {
          statusLine.className = "status-line done";
          statusLine.innerHTML = `<span class="check">&#10003;</span><span>Retrieved ${data.chunks} chunks in ${data.elapsed_s}s</span>`;
        } else if (event === "delta") {
          answerText += data.text;
          renderAnswerParagraphs(answerEl, answerText);
        } else if (event === "final") {
          const wallMs = performance.now() - t0;
          const turn = { query, role, partner, response: data };
          history.push(turn);
          renderHistory();
          composer.querySelector("button").disabled = false;
          return;
        }
      }
    }
  } catch (err) {
    statusLine.className = "status-line done";
    statusLine.innerHTML = `<span style="color:var(--red)">Request failed: ${esc(err.message)}</span>`;
  }
  composer.querySelector("button").disabled = false;
}

// Small, safe markdown subset for the LLM's answers: paragraphs, "- " bullet
// lists, "#"/"##" headings, ```fenced code blocks```, and **bold** labels --
// the formatting llm/generate.py's prompt actually produces, extended to
// include fenced code once the prompt started allowing the model to
// translate documented facts into working code (see SYSTEM_PROMPT's
// "translate those facts into a different format" paragraph) -- without
// this, a returned code sample would render as one run-on paragraph with
// the ``` fence markers left in as visible text and all indentation
// collapsed. Deliberately NOT a full markdown renderer and NEVER uses
// innerHTML with model/user text: every node below is built with
// createElement + textContent, so nothing the model (or a user asking it to
// "repeat back <script>...") writes can execute as markup, even though this
// function's whole job is to turn text into formatted HTML.
const isBulletLine = (l) => /^[-*]\s+/.test(l);
const FENCE_RE = /```(\w*)\n?([\s\S]*?)```/g;

function renderAnswerParagraphs(el, text) {
  el.textContent = "";
  let lastIndex = 0;
  FENCE_RE.lastIndex = 0;
  let m;
  while ((m = FENCE_RE.exec(text)) !== null) {
    renderTextBlocks(el, text.slice(lastIndex, m.index));
    const pre = document.createElement("pre");
    pre.className = "code-block";
    const code = document.createElement("code");
    if (m[1]) code.className = `lang-${m[1]}`;
    code.textContent = m[2].replace(/\n$/, "");
    pre.appendChild(code);
    el.appendChild(pre);
    lastIndex = FENCE_RE.lastIndex;
  }
  // Whatever's left after the last fence -- all of it, if the model hasn't
  // written any code -- or an still-open fence mid-stream (the closing ```
  // hasn't arrived yet): rendered as plain text blocks either way, same as
  // any other in-progress line while streaming.
  renderTextBlocks(el, text.slice(lastIndex));
}

function renderTextBlocks(el, text) {
  for (const block of text.split(/\n\n+/)) {
    if (!block.trim()) continue;
    const lines = block.split("\n").filter((l) => l.trim());

    const headingMatch = /^#{1,6}\s+(.*)$/.exec(lines[0]);
    if (headingMatch) {
      const h = document.createElement("p");
      h.className = "ans-heading";
      appendInline(h, headingMatch[1]);
      el.appendChild(h);
      if (lines.length > 1) renderTextBlocks(el, lines.slice(1).join("\n"));
      continue;
    }

    // The model sometimes leads a list with a plain or **bold** category
    // line before the "- " items (e.g. "**Sales APIs**\n- Get...\n- Export...")
    // instead of putting the whole block in dashes -- split off any such
    // leading non-bullet lines so the rest can still render as a real <ul>.
    let split = 0;
    while (split < lines.length && !isBulletLine(lines[split])) split++;
    const headerLines = lines.slice(0, split);
    const bulletLines = lines.slice(split);
    const isList = bulletLines.length > 0 && bulletLines.every(isBulletLine) && split < lines.length;

    if (isList) {
      for (const h of headerLines) {
        const p = document.createElement("p");
        appendInline(p, h);
        el.appendChild(p);
      }
      const ul = document.createElement("ul");
      for (const line of bulletLines) {
        const li = document.createElement("li");
        appendInline(li, line.replace(/^[-*]\s+/, ""));
        ul.appendChild(li);
      }
      el.appendChild(ul);
    } else {
      const p = document.createElement("p");
      lines.forEach((line, i) => {
        if (i > 0) p.appendChild(document.createElement("br"));
        appendInline(p, line);
      });
      el.appendChild(p);
    }
  }
}

// Splits `**bold**` and `` `inline code` `` spans out of `text` and appends
// each piece to `container` as a <strong>, a <code>, or a plain text node --
// the only inline markup recognized, matching what the model's prompt
// actually asks it to produce (inline code spans became common once the
// prompt started allowing documented values -- endpoints, header/field
// names -- to appear inline in prose, not just inside fenced code blocks).
function appendInline(container, text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    const bold = /^\*\*([^*]+)\*\*$/.exec(part);
    const code = /^`([^`]+)`$/.exec(part);
    if (bold) {
      const strong = document.createElement("strong");
      strong.textContent = bold[1];
      container.appendChild(strong);
    } else if (code) {
      const el = document.createElement("code");
      el.className = "inline-code";
      el.textContent = code[1];
      container.appendChild(el);
    } else if (part) {
      container.appendChild(document.createTextNode(part));
    }
  }
}

// ── Replay: renders every finished turn from `history`, called after each
// new answer completes and whenever the Debug toggle changes. One code path
// for "just streamed" and "toggled debug on for an old turn," matching how
// app/streamlit_app.py's script reran top-to-bottom on every interaction. ──
function renderHistory() {
  logEl.innerHTML = "";
  for (const turn of history) {
    renderStaticTurn(turn);
  }
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function renderStaticTurn(turn) {
  const { query, role, partner, response: r } = turn;

  const userTurn = document.createElement("div");
  userTurn.className = "turn user";
  userTurn.innerHTML = `<div class="avatar">🧑</div><div class="card"></div>`;
  const userCard = userTurn.querySelector(".card");
  const b = document.createElement("b");
  b.textContent = `${role} @ ${partner}: `;
  userCard.appendChild(b);
  userCard.appendChild(document.createTextNode(query));
  logEl.appendChild(userTurn);

  const assistantTurn = document.createElement("div");
  assistantTurn.className = "turn assistant";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🤖";
  const card = document.createElement("div");
  card.className = "card";
  assistantTurn.appendChild(avatar);
  assistantTurn.appendChild(card);

  const tags = [];
  if (r.abstained) tags.push(["abstained", "abstained"]);
  if (r.permission_refused) tags.push(["refused", "permission refused"]);
  if (r.degraded_rerank) tags.push(["degraded", "degraded.rerank"]);
  if (tags.length) {
    const tagsEl = document.createElement("div");
    tagsEl.className = "tags";
    for (const [cls, label] of tags) {
      const t = document.createElement("span");
      t.className = `tag ${cls}`;
      t.textContent = label;
      tagsEl.appendChild(t);
    }
    card.appendChild(tagsEl);
  }

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  renderAnswerParagraphs(answerEl, r.answer || "");
  card.appendChild(answerEl);

  if (r.citations && r.citations.length) {
    const cap = document.createElement("div");
    cap.className = "meta-caption";
    cap.textContent = "Citations: " + r.citations.join(", ");
    card.appendChild(cap);
  }

  if (r.rewritten_query) {
    card.appendChild(buildExpander(
      `Rewritten query (${(r.rewrite_rules_applied || []).join(", ")})`,
      (body) => { body.textContent = r.rewritten_query; },
    ));
  }

  if (r.scores && r.scores.length) {
    card.appendChild(buildExpander(`Retrieval scores (${r.scores.length})`, (body) => {
      const table = document.createElement("table");
      table.className = "scores";
      table.innerHTML = "<thead><tr><th>Doc</th><th>Heading</th><th class='num'>Rerank</th><th class='num'>Fused</th><th>Arms</th></tr></thead>";
      const tbody = document.createElement("tbody");
      for (const s of r.scores) {
        const tr = document.createElement("tr");
        const cells = [
          s.doc_id, s.heading,
          s.rerank_score != null ? s.rerank_score.toFixed(4) : "-",
          s.fused_score != null ? s.fused_score.toFixed(4) : "-",
          (s.arms || []).join(", "),
        ];
        cells.forEach((v, i) => {
          const td = document.createElement("td");
          if (i === 2 || i === 3) td.className = "num";
          td.textContent = v;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      body.appendChild(table);
    }));
  }

  if (r.timings_ms) {
    card.appendChild(buildExpander("Timings", (body) => {
      const pre = document.createElement("pre");
      pre.className = "json";
      pre.textContent = JSON.stringify(r.timings_ms, null, 2);
      body.appendChild(pre);
    }));
  }

  if (debugToggle.checked && r.trace && r.trace.length) {
    const section = document.createElement("div");
    section.className = "debug-section";
    const cap = document.createElement("div");
    cap.className = "debug-caption";
    cap.textContent = "Debug mode — pipeline flowchart for this query";
    section.appendChild(cap);
    section.appendChild(buildFlowchart(r.trace));
    section.appendChild(buildTimeline([["Timeline", r.trace]]));
    section.appendChild(buildDetailGrid(r.trace));
    card.appendChild(section);
  }

  logEl.appendChild(assistantTurn);
}

function buildExpander(title, fillBody) {
  const details = document.createElement("details");
  details.className = "expander";
  const summary = document.createElement("summary");
  summary.textContent = title;
  const body = document.createElement("div");
  body.className = "expander-body";
  fillBody(body);
  details.appendChild(summary);
  details.appendChild(body);
  return details;
}

// ── Flowchart + timeline: ported line-for-line from app/streamlit_app.py's
// build_flow_svg()/build_timeline_html() (Python), so Debug mode reads
// exactly the way it did there. Kept in JS now instead of asking the server
// to render SVG strings, since this is a real page with no iframe sandbox to
// work around. ──

const NODE_COLORS = {
  "Query Rewrite": "#1c2b42", "Embed Query": "#c15d52",
  "Keyword Search": "#1c2b42", "Semantic Search": "#1c2b42",
  "RRF Fusion": "#5c6b80", "Rerank (competitive)": "#c15d52",
  "Guaranteed Fetch": "#2f6690", "Rerank (guaranteed)": "#2f6690",
  "Confidence Gate": "#b0413a", "Generate Answer": "#c15d52",
};
const NODE_SUBTITLES = { "Rerank (competitive)": "Cross-Encoder Rerank" };
const NODE_POS = {
  "Query Rewrite": [130, 1],
  "Embed Query": [420, 0], "Keyword Search": [420, 1], "Guaranteed Fetch": [420, 2],
  "Semantic Search": [710, 0], "Rerank (guaranteed)": [710, 2],
  "RRF Fusion": [1000, 0.5],
  "Rerank (competitive)": [1290, 0.5],
  "Confidence Gate": [1580, 1],
  "Generate Answer": [1870, 1],
};
const FLOW_EDGES = [
  ["Query Rewrite", "Embed Query"], ["Query Rewrite", "Keyword Search"],
  ["Embed Query", "Semantic Search"],
  ["Keyword Search", "RRF Fusion"], ["Semantic Search", "RRF Fusion"],
  ["RRF Fusion", "Rerank (competitive)"],
  ["Guaranteed Fetch", "Rerank (guaranteed)"],
  ["Rerank (competitive)", "Confidence Gate"], ["Rerank (guaranteed)", "Confidence Gate"],
  ["Confidence Gate", "Generate Answer"],
];
const BOX_W = 240, LANE_GAP = 40, BATCH_COLOR = "#f0a030", MAX_INLINE_VALUE_LEN = 18;

function wrapText(text, maxChars) {
  const words = text.split(" ");
  const lines = [];
  let current = "";
  for (const w of words) {
    const candidate = (current + " " + w).trim();
    if (candidate.length > maxChars && current) {
      lines.push(current);
      current = w;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function nodeLabelLine(key, value) {
  if (Array.isArray(value)) return `${key}: ${value.length}`;
  const s = String(value);
  if (s.length > MAX_INLINE_VALUE_LEN) return key;
  return `${key}: ${s}`;
}

function traceOrigin(trace) {
  const starts = trace.map((s) => s.started_at).filter((v) => v != null);
  return starts.length ? Math.min(...starts) : null;
}

function startOffsetMs(step, origin) {
  if (step.started_at == null || origin == null) return null;
  return Math.round((step.started_at - origin) * 1000);
}

function buildFlowchart(trace) {
  const wrap = document.createElement("div");
  wrap.className = "flow-wrap";
  const byName = {};
  for (const s of trace) byName[s.name] = s;
  const numbers = {};
  PIPELINE_STEP_ORDER.forEach((n, i) => (numbers[n] = i + 1));
  const origin = traceOrigin(trace);

  const content = {};
  for (const [name, step] of Object.entries(byName)) {
    const titleLines = wrapText(`${numbers[name]}. ${name.toUpperCase()}`, 22);
    const bodyLines = [];
    if (NODE_SUBTITLES[name]) bodyLines.push(NODE_SUBTITLES[name]);
    for (const [k, v] of Object.entries(step.outputs || {})) bodyLines.push(nodeLabelLine(k, v));
    const start = startOffsetMs(step, origin);
    bodyLines.push(`⏱ ${step.timing_ms} ms` + (start != null ? ` · starts +${start} ms` : ""));
    content[name] = [titleLines, bodyLines];
  }

  const neededHeight = (title, body) => 16 + 17 * title.length + 8 + 18 * body.length + 8;
  const boxH = Math.max(...Object.values(content).map(([t, b]) => neededHeight(t, b)));
  const lanePitch = boxH + LANE_GAP;
  const pad = 20;
  const centerY = (name) => pad + boxH / 2 + NODE_POS[name][1] * lanePitch;

  const names = Object.keys(byName);
  const maxX = Math.max(...names.map((n) => NODE_POS[n][0])) + BOX_W / 2 + 40;
  const maxY = Math.max(...names.map((n) => centerY(n))) + boxH / 2 + pad;

  const svg = [`<svg viewBox="0 0 ${maxX} ${maxY.toFixed(0)}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">`];
  svg.push(`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9aa5b1"/></marker></defs>`);

  for (const [a, b] of FLOW_EDGES) {
    if (byName[a] && byName[b]) {
      const x1 = NODE_POS[a][0] + BOX_W / 2, y1 = centerY(a);
      const x2 = NODE_POS[b][0] - BOX_W / 2, y2 = centerY(b);
      const mx = Math.floor((x1 + x2) / 2);
      svg.push(`<path d="M${x1},${y1.toFixed(0)} C${mx},${y1.toFixed(0)} ${mx},${y2.toFixed(0)} ${x2},${y2.toFixed(0)}" fill="none" stroke="#9aa5b1" stroke-width="2.5" marker-end="url(#arrow)"/>`);
    }
  }

  for (const [name, step] of Object.entries(byName)) {
    const [titleLines, bodyLines] = content[name];
    const cx = NODE_POS[name][0], cy = centerY(name);
    const color = NODE_COLORS[name] || "#1c2b42";
    const top = cy - boxH / 2, left = cx - BOX_W / 2;
    const outline = step.outputs && step.outputs.batched_call
      ? ` stroke="${BATCH_COLOR}" stroke-width="4" stroke-dasharray="9 5"` : "";

    svg.push(`<g class="trace-node" tabindex="0">`);
    svg.push(`<rect x="${left}" y="${top.toFixed(0)}" width="${BOX_W}" height="${boxH.toFixed(0)}" rx="14" fill="${color}"${outline}/>`);
    let y = top + 26;
    svg.push(`<text x="${left + 16}" y="${y.toFixed(0)}" fill="white" font-size="14" font-weight="700">`);
    titleLines.forEach((line, i) => svg.push(`<tspan x="${left + 16}" dy="${i === 0 ? 0 : 17}">${esc(line)}</tspan>`));
    svg.push(`</text>`);
    y += 17 * (titleLines.length - 1) + 24;
    svg.push(`<text x="${left + 16}" y="${y.toFixed(0)}" fill="#e8ecf1" font-size="12">`);
    bodyLines.forEach((line, i) => svg.push(`<tspan x="${left + 16}" dy="${i === 0 ? 0 : 18}">${esc(line)}</tspan>`));
    svg.push(`</text>`);
    svg.push(`</g>`);
  }

  svg.push("</svg>");
  wrap.innerHTML = svg.join("\n");
  return wrap;
}

function niceTick(axisMaxMs) {
  for (const tick of [100, 200, 500, 1000, 2000, 5000, 10000]) {
    if (axisMaxMs / tick <= 8) return tick;
  }
  return 20000;
}

function timelineRows(trace) {
  const origin = traceOrigin(trace);
  if (origin == null) return { rows: [], total: 0 };
  const rows = [];
  for (const name of PIPELINE_STEP_ORDER) {
    const step = trace.find((s) => s.name === name);
    if (!step || step.started_at == null) continue;
    const start = (step.started_at - origin) * 1000;
    rows.push([name, start, step.timing_ms, !!(step.outputs && step.outputs.batched_call)]);
  }
  const total = rows.reduce((m, [, start, dur]) => Math.max(m, start + dur), 0);
  return { rows, total };
}

function buildTimeline(sections) {
  const wrap = document.createElement("div");
  const built = sections.map(([title, trace]) => [title, ...Object.values(timelineRows(trace))]);
  const axisMax = Math.max(...built.map(([, , total]) => total), 0) * 1.03 || 1;
  const tick = niceTick(axisMax);
  const ticks = [];
  for (let t = 0; t <= axisMax; t += tick) ticks.push(t);
  const numbers = {};
  PIPELINE_STEP_ORDER.forEach((n, i) => (numbers[n] = i + 1));

  const grid = () => ticks.map((t) => `<i style="left:${((t / axisMax) * 100).toFixed(2)}%"></i>`).join("");

  let html = "";
  for (const [title, rows, total] of built) {
    html += `<div class="timeline"><h4>${esc(title)} <span>— ended at ${(total / 1000).toFixed(1)} s</span></h4>`;
    const labels = ticks.map((t) => `<b style="left:${((t / axisMax) * 100).toFixed(2)}%">${(t / 1000)}s</b>`).join("");
    html += `<div class="tl-axis"><span></span><div class="scale">${labels}</div><span></span></div>`;
    for (const [name, start, dur, batched] of rows) {
      const color = NODE_COLORS[name] || "#1c2b42";
      const border = batched ? `outline:2px dashed ${BATCH_COLOR};` : "";
      html += `<div class="tl-row"><span>${numbers[name]}. ${esc(name)}</span>` +
        `<div class="tl-track">${grid()}<div class="tl-bar" style="left:${((start / axisMax) * 100).toFixed(2)}%;` +
        `width:${((dur / axisMax) * 100).toFixed(2)}%;background:${color};${border}"></div></div>` +
        `<span class="val">${dur.toLocaleString(undefined, { maximumFractionDigits: 0 })} ms</span></div>`;
    }
    html += "</div>";
  }
  html += `<div class="tl-legend"><span style="color:${BATCH_COLOR}">▭ dashed orange</span> = these steps were one shared model call. A bar's left edge is when the step started; steps whose bars overlap ran at the same time.</div>`;
  wrap.innerHTML = html;
  return wrap;
}

function buildDetailGrid(trace) {
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  const numbers = {};
  PIPELINE_STEP_ORDER.forEach((n, i) => (numbers[n] = i + 1));
  for (const step of trace) {
    const card = document.createElement("div");
    card.className = "detail-card";
    const h5 = document.createElement("h5");
    h5.textContent = `${numbers[step.name] || "?"}. ${step.name}`;
    const timing = document.createElement("div");
    timing.className = "timing";
    timing.textContent = `⏱ ${step.timing_ms} ms`;
    card.appendChild(h5);
    card.appendChild(timing);
    card.appendChild(buildExpander("Inputs", (body) => {
      const pre = document.createElement("pre");
      pre.className = "json";
      pre.textContent = JSON.stringify(step.inputs, null, 2);
      body.appendChild(pre);
    }));
    card.appendChild(buildExpander("Outputs", (body) => {
      const pre = document.createElement("pre");
      pre.className = "json";
      pre.textContent = JSON.stringify(step.outputs, null, 2);
      body.appendChild(pre);
    }));
    if (step.detail != null) {
      card.appendChild(buildExpander("Full detail", (body) => {
        const pre = document.createElement("pre");
        pre.className = "json";
        pre.textContent = JSON.stringify(step.detail, null, 2);
        body.appendChild(pre);
      }));
    }
    grid.appendChild(card);
  }
  return grid;
}
