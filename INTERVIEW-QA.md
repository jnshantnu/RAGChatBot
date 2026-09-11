# Interview Q&A — Pipeline Internals (from Debug Mode walkthrough)

Companion to `RAG-Assistant-Interview-Prep.docx` (demo script + high-level architecture Q&A). This file covers the deeper, mechanism-level questions that came up while walking through the Debug Mode flowchart step by step.

---

### What does `embedding_dims: 1536` mean?

It's the size of the vector produced by the embedding step — not the text itself, just how many numbers represent it. `qwen/qwen3-embedding-8b`'s native output is actually 4096 dimensions, but pgvector's HNSW index caps at 2000, so `ingest/embed.py` requests a truncated 1536-dim version instead (Matryoshka representation — a trained model whose truncated prefix is still meaningful, not garbage). This number has to match the `vector(1536)` column width in `db/schema.sql`, or ingestion breaks.

### Where is `OVER_FETCH = 50` defined, and why that number?

`retrieval/hybrid_search.py:16`. It's how many candidates each search arm (keyword, semantic) pulls before fusion — deliberately more than the ~5 that end up cited, so RRF fusion and reranking have a wide enough pool to work with. Fetching only 5 per arm up front would silently cap recall before fusion even gets a chance to combine the two arms' rankings.

### How do I read the keyword-search score vs. the semantic-search score?

They're opposite directions and not comparable. Keyword search's score is `ts_rank_cd` (Postgres full-text relevance) — **higher is better**, no fixed ceiling. Semantic search's score is **cosine distance**, not similarity — **lower is better** (0 = identical direction). You can't compare `3` against `0.19` directly; trust the *order* of each list (index 0 = that arm's best pick), not the raw numbers.

### How do I read the RRF `fused_score`?

Formula: `1/(60+rank)` per arm, summed if a chunk appears in both. The `60` (`K`) softens the gap between rank 1 and rank 2 so one arm's confident pick can't dominate. Practical rule: the best possible score from *one* arm alone is `1/61 ≈ 0.0164` — so **any fused score above ~0.0164 proves both arms found it**; below that, only one arm did. (The `arms` field in the trace now shows this directly instead of requiring the math.)

### Explain RRF in one sentence.

Like two judges scoring a competition on different, incompatible scales (100-point technical score vs. 10-point artistic score) — you can't add their raw scores, but you can combine based on *where* they each ranked the same contestant. RRF does that: it fuses on rank position, never on the original score value.

### Why does cross-encoder reranking happen *after* RRF fusion, not instead of it?

Cost vs. accuracy funnel — like a hiring pipeline: cheap resume screening (keyword + semantic search, done on everything) narrows the field before the expensive, accurate interview (the cross-encoder) only has to judge the shortlist. The cross-encoder reads query and chunk *together* — far more accurate than comparing two independently-computed signals — but too slow to run on the whole corpus. RRF's job is only to produce a good-enough shortlist (top 20) cheaply; the reranker's job is to judge that shortlist precisely.

### Step 6 "Rerank (competitive)" and Step 8 "Rerank (guaranteed)" — same thing?

Same function, same cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `retrieval/rerank.py`). The name difference describes *which candidate pool* is being scored: step 6 scores the top 20 that had to win a spot through keyword+semantic+fusion; step 8 scores the small guaranteed-content pool that never had to compete for one.

### What does `degraded: False` mean on the rerank step?

A health flag, not a relevance score. If the cross-encoder model fails to load or throws an exception, `retrieval/pipeline.py` catches it and falls back to plain RRF order instead of crashing — `degraded: True` marks that this happened, and the confidence gate switches to a weaker fallback logic in that case (raw cosine similarity) instead of trusting a score that was never actually computed.

### Why do Steps 7 & 8 (Guaranteed Fetch / Rerank) exist?

A real regression, not a hypothetical safety net: `ADSK-BIZ-RULES.md` (8 chunks) is small enough that it could — and once did — lose the top-20 rerank competition against 1,000+ other chunks for certain phrasings, silently vanishing from the answer even though it's supposed to inform every relevant response. Fix: fetch every `guaranteed: true` chunk directly from Postgres, unconditionally, every time (step 7); rerank *only that small set* (step 8) — not to decide inclusion (all are always included), only to tell the confidence gate how relevant it actually is to this specific question, since that score can rescue an otherwise-weak answer.

### Where does `ADSK-BIZ-RULES.md` actually live — disk or Postgres?

Both, at different times. The source markdown file lives on disk; ingestion reads it once, chunks it, embeds it, and writes everything into the same `chunks` table every other document lives in. Every live query reads from Postgres — the file on disk is irrelevant after ingestion. Verified directly: `SELECT ... FROM chunks WHERE metadata->>'guaranteed' = 'true'` returns exactly 8 rows, one per `##` section of the file.

### Explain Step 9, the Confidence Gate.

Decides whether to answer at all, *before* the (expensive) LLM call — reads step 6's and step 8's best scores. Three checks, in order: (1) if the top competitive score clears `0.20`, answer (`reason: "ok"`); (2) if not, but the top score is far ahead of the runner-up (10x+, both above a small floor), still answer — a real match can score low on vague phrasing but still stand out from the noise around it (`"ok_dominant_top1"`); (3) if not, but the guaranteed-content score alone clears `0.20`, still answer (`"ok_guaranteed_content"`) — some questions are only answerable from business-rules content, not the competitive corpus. Otherwise: abstain, no LLM call at all.

### Is `MIN_RERANK_SCORE = 0.20` a literal "20% probability"?

No — don't over-read the sigmoid. It's a confidence score bounded 0–1 (higher = more confident), not a calibrated probability (the model was never trained/scaled so that "0.20" corresponds to a real 20% hit rate). The `0.20` cutoff itself was found empirically, not derived mathematically: a real correct-but-vague answer scored `0.23`; real noise scored `0.0017`–`0.0043`. The threshold just marks where signal and noise happened to separate for this model and this corpus.

### Which models does this project actually use, and where?

1. **`qwen/qwen3-embedding-8b`** (OpenRouter, network call) — `ingest/embed.py`, both at ingest time (`embed_texts`) and query time (`embed_query`, step 2). Powers semantic search.
2. **`openai/gpt-5.6-luna`** (OpenRouter, network call, set in `.env` — not `config.py`'s fallback default) — `llm/generate.py`'s `generate_answer()`, step 10. Writes the final answer.
3. **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (local, `sentence-transformers`, runs on this VPS's own 4 CPU cores — no network call) — `retrieval/rerank.py`, steps 6 and 8. The only model actually running on this box.

### Why is the local cross-encoder reranker slow?

Two compounding reasons: (1) no GPU — this VPS's 4 CPU cores do sequentially what a GPU farm would do in parallel across thousands of cores; (2) no caching is possible for this step, unlike embeddings (each document chunk is embedded once, ever, and reused forever) — the reranker has to read *this exact query* together with *each* candidate's full text, fresh, every single query, because that comparison is different every time.

### If paying for it is an option, what are the alternatives to the local reranker?

- **Hosted reranking APIs** — Cohere Rerank, Jina Reranker, Voyage AI Rerank. Same core idea (cross-encoder scoring) on GPU-backed infrastructure instead of local CPU. Notably, the original case study's own reference architecture names Cohere Rerank / BGE-reranker-v2-m3 directly — this isn't a deviation from the reference design, it's what it assumed.
- **Same model, paid GPU hosting** — Modal, Replicate, Baseten, RunPod serverless GPU, running the identical model off-box.
- **LLM-as-reranker** — have the already-paid chat model score/rank candidates via structured prompting instead of a dedicated reranker model. Fewer moving parts, but usually slower per call than a purpose-built reranking API.
- **Trade-off worth naming out loud:** the embedding and generation models are already hosted/paid. The reranker is the *only* model currently running on this project's own infrastructure — moving it off-box means nothing in the system runs locally anymore, which is a real trade, not a free win.

### Why does Step 10 (Generate Answer) take ~5 seconds?

Different mechanism than the embedding step's slowness. The model has to (1) first read the entire prompt — system instructions plus every citable chunk (up to 5 competitive + up to 8 guaranteed) — before generating anything, and (2) write the answer **token by token**, autoregressively — each token requires a full pass through the whole model, and the next token can't start until the previous one finishes. A ~400-character answer is 100+ sequential steps. This runs on OpenRouter's infrastructure, not local hardware, so the only real levers are: a faster/smaller model, a shorter answer, less context in the prompt, or streaming (which doesn't reduce the total 5s, but shows tokens as they arrive instead of one long wait — usually the bigger practical win).

### Where do precision, recall, and F1 apply in this project?

- **Retrieval recall — already measured.** `eval/run_eval.py`'s `recall@5`/`recall@10`: of all answerable golden-set questions, what fraction had the correct document show up in the top 5/10 results?
- **Retrieval precision — a real, acknowledged gap.** Never measured — the eval only checks whether the *one* expected document appeared somewhere in the top 5; it doesn't penalize the other 4 slots being irrelevant. The golden set's structure (one expected doc per question, not exhaustively labeled) makes this hard to compute as-is.
- **The confidence gate is secretly a binary classifier** (abstain vs. answer), and precision/recall apply directly: *recall* on the "should abstain" class is exactly `"retrieval-gate abstain rate on unanswerable"` in the eval output; *precision* on that same class (of all the times it abstained, how many were actually correct, vs. wrongly refusing a good question) is **not currently tracked** — another honest gap.
- **`MIN_RERANK_SCORE` is the literal decision threshold trading one against the other.** Lowering it from `0.30` to `0.20` earlier in this project directly increased how often genuinely answerable-but-vague questions get answered, at the cost of risking more noise slipping through — the textbook classifier-threshold trade-off, tuned against real labeled examples (the golden set), not picked arbitrarily.
- **F1** would be the single number balancing those two abstain-classifier metrics — useful because neither extreme (always abstain / never abstain) is good, and optimizing only one of precision or recall in isolation rewards exactly that kind of useless extreme.
- **Generation/citation precision**, more loosely: `llm/generate.py`'s `uncited_claims_flagged` is a crude proxy for "of the claims made, how many are actually backed by a real citation" — the same underlying question precision asks, just not measured as a real rate today.
