# RAG Chatbot POC

A scaled-down, hands-on rebuild of the partner-support RAG architecture from
the case study: Postgres + pgvector, hybrid retrieval (keyword GIN + semantic
HNSW), RRF fusion, cross-encoder rerank, ACL-aware permission filtering, a
confidence gate that abstains instead of hallucinating, and a small eval loop.

Knowledge base: 18 real Autodesk Partner WebServices reference PDFs (BuySell
+ NxM business models) -> 838 chunks, in `RAG-KB-Documents/`.

## Setup

1. Postgres 14 + pgvector are already installed locally; `db/schema.sql` has
   been applied to the `ragchatbot` database (role `ragapp`).
2. `python3 -m venv venv && ./venv/bin/pip install -r requirements.txt` (already done).
3. Copy `.env.example` to `.env` (already done) and fill in `OPENROUTER_API_KEY`
   from https://openrouter.ai/keys.
4. Embeddings: `qwen/qwen3-embedding-8b` via OpenRouter, requested at 1536
   dims (native is 4096, Matryoshka-truncatable; pgvector's HNSW index caps at
   2000 dims for the standard `vector` type -- see `ingest/embed.py`). Queries
   get an instruction prefix the model expects for retrieval; documents don't
   (asymmetric embedding, see `OPENROUTER_EMBEDDING_QUERY_INSTRUCTION` in `.env`).

## Build order (matches the case study's data flow)

| Phase | What | Files |
|---|---|---|
| 0 | Environment: Postgres + pgvector (built from source), venv | `db/schema.sql` |
| 1 | Real corpus: 18 Autodesk Partner WebServices PDFs | `RAG-KB-Documents/*.pdf` |
| 2 | Schema: one `chunks` table, GIN + HNSW indexes, ACL array | `db/schema.sql` |
| 3 | Ingestion: PDF page-chunk -> embed (OpenRouter) -> upsert, idempotent | `ingest/*.py` |
| 4 | Hybrid retrieval + RRF fusion + cross-encoder rerank + confidence gate | `retrieval/*.py` |
| 5 | Context assembly + LLM generation + citation check | `llm/generate.py` |
| 6 | Orchestration incl. pre-retrieval permission refusal | `chat.py` |
| 7 | Minimal web UI | `app/main.py`, `app/static/index.html` |
| 8 | Eval loop: recall@5/@10, abstain correctness | `eval/*.py` |
| 9 | Lightweight observability (JSON request log) | `logs/requests.jsonl` (written by `app/main.py`) |

## Running it

```bash
source venv/bin/activate

# 1. Ingest the corpus (idempotent -- re-run any time docs change)
python -m ingest.ingest

# 2. Inspect raw retrieval before any LLM is involved
python -m retrieval.cli_test "How do I authenticate to the Partner WebServices API?"
python -m retrieval.cli_test "Does the platform support Slack notifications?"   # should abstain

# 3. Run the eval harness
python -m eval.run_eval

# 4. Start the web UI
uvicorn app.main:app --reload
# open http://localhost:8000
```

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via
`sentence-transformers`) loads once per process (~8s) and is then reused for
every subsequent request in that process (~0.5s/query) -- the first query
after starting `uvicorn` will be slower than the rest.

## Three canonical test questions

1. **Answerable**: "How do I authenticate to the Partner WebServices API?" -> cited answer from the auth guide.
2. **Out-of-corpus**: "Does the platform support Slack notifications for order status changes?" -> abstain ("I don't have that information.") before any LLM call.
3. **Permission-refusal path** (currently dormant, see below): a query naming a restricted category with a role that lacks it -> refused before retrieval runs at all.

## Current ACL status: mostly dormant

The real KB is presently ingested as **all public** (18 PDFs, technical
reference docs). The one document identified as genuinely restricted --
`VAR dynamic Pricelist V2.xlsm` (a pricing spreadsheet) -- is a spreadsheet,
not a PDF, and isn't ingested yet (spreadsheet chunking doesn't work like
prose; it's a follow-up exercise, see below). That means:

- The role/partner selectors in the UI and `chat.py`'s ACL machinery
  (`acl && user_groups` at the SQL level, the pre-retrieval permission-refusal
  keyword check) are fully wired and tested (originally verified against a
  synthetic ACME/Globex/principal-commission corpus during development), but
  currently have nothing restricted in the live KB to actually gate.
- To bring the ACL demo back to life: ingest the pricelist (needs a
  spreadsheet loader, e.g. `openpyxl`) tagged `acl: [role:principal]`, the
  same way `partner_incentives.md` worked in the original synthetic corpus.

## What broke, and what fixed it, along the way

Three real engineering findings from this build, all documented in the mind
map artifact in detail:

1. **Postgres's default keyword search ANDs every query term** -- a single
   off-corpus word zeroed out otherwise-perfect matches. Fixed by rewriting
   `plainto_tsquery`'s `&` into `|` (`retrieval/hybrid_search.py`).
2. **Switching the embedding model (OpenAI -> Qwen3-Embedding-8B) broke the
   confidence gate.** Raw cosine similarity from Qwen3 didn't separate
   relevant from irrelevant results on this corpus -- an unanswerable query
   ("Can I integrate with Salesforce?") scored *higher* similarity than a
   genuinely relevant match, because bi-encoder similarity picks up shared
   vocabulary/topic, not actual relevance. Fixed by implementing the
   cross-encoder rerank step the case study's own diagram marks as the
   missing piece (`retrieval/rerank.py`), and pointing the confidence gate at
   its calibrated 0-1 relevance score instead
   (`retrieval/confidence.py:MIN_RERANK_SCORE`). The raw-similarity gate is
   kept as a documented `degraded.rerank` fallback only.
3. **A chunk can't be retrieved for identity it never states.** "which apis
   are available for buy sell model" scored 0.03 (abstain) even though the
   right page was retrieved at rank 1 -- because the word "BuySell" never
   appears anywhere in that PDF's actual text, only in its filename. Neither
   the embedding, the keyword index, nor the reranker had any way to connect
   the query's "buy sell" to a page that never says it. Fixed by prefixing
   every chunk with a filename-derived document title (`[BuySell PWS Place
   Order API Implementation Guide]`) before it's embedded/indexed/reranked --
   standard "contextual retrieval," and the fix specifically had to come from
   the filename, not the PDF's own content, since the content itself never
   states the business model either (see `ingest/chunker.py:chunk_pdf` and
   `ingest/ingest.py:_doc_title`). Added as a permanent regression case in
   `eval/golden_set.json`.

## What's intentionally simplified vs. the case study

- **Observability**: one JSON line per request in `logs/requests.jsonl`
  instead of OpenTelemetry spans -> Langfuse. Same information (timings,
  scores, degraded/abstain flags), no tracing infra.
- **Query rewriter**: not implemented -- queries go straight to embedding.
- **Reindexing / blue-green**: not implemented -- ingestion re-embeds in place.
- **Cache lookup**: not implemented (query + index_version + permission key).
- **Spreadsheet/PPTX ingestion**: not implemented -- PDFs only for now (see
  ACL section above for why this matters for the pricelist specifically).
- **Confidence gate's degraded fallback thresholds**
  (`retrieval/confidence.py: MIN_SEMANTIC_SIMILARITY`/`BORDERLINE_SIMILARITY`)
  are illustrative constants tuned by eye, not a calibrated model -- only used
  when the reranker itself fails to load.
