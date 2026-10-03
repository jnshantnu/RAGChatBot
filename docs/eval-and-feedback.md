# Evaluation and the feedback loop

Three layers, each catching a different kind of failure. The first two answer
"is the system working on the questions we thought of." The third is what
makes it **improve with time**, not just stay flat: real usage feeds new,
real failures back into the first two, so the eval sets grow from evidence
instead of staying fixed at whatever someone happened to write down first.

```
   fixed eval sets                    real usage
  ┌─────────────────┐          ┌──────────────────────┐
  │ golden_set.json  │          │  chat UI              │
  │ (retrieval)      │          │   -> 👍/👎 + comment   │
  │ generation_eval_ │          │   -> logs/feedback     │
  │  set.json        │          │      .jsonl            │
  │ (answer quality) │          │  every request already │
  └────────┬─────────┘          │  logs to               │
           │ run before         │  logs/requests.jsonl   │
           │ shipping a change  └──────────┬─────────────┘
           ▼                               ▼
     pass/fail report          eval/triage_feedback.py
                                (joins the two logs by
                                 request_id, prints full
                                 context for review)
                                               │
                                               ▼
                                a real failure becomes a
                                new row in one of the eval
                                sets on the left -- the loop
                                closes
```

## Layer 1 — Retrieval quality (`eval/run_eval.py`)

Did the right *document* get found? recall@5/@10 against `eval/golden_set.json`
(20 questions), plus abstain correctness on the deliberately unanswerable
ones. No LLM call -- fast, and the first thing to run after any change to
chunking, embeddings, or the retrieval pipeline.

```bash
python -m eval.run_eval            # baseline pipeline
python -m eval.run_eval parallel   # optimized pipeline
```

Query-understanding accuracy (`eval/run_understanding_eval.py`) and the LLM
classifier/rewrite fallbacks are their own eval scripts under this same
layer -- see `docs/query-understanding.md`.

## Layer 2 — Generation quality (`eval/run_generation_eval.py`)

Retrieval can find the perfect chunk and the answer can still be wrong --
this is a genuinely different failure mode, and nothing else in this repo
checked it before this file existed. Runs the FULL pipeline (real Postgres,
real embedding API, real LLM) and checks the actual answer text against
`eval/generation_eval_set.json`:

* **`expected_facts`** -- every listed fact must appear (case-insensitive
  substring match) in the answer. Deliberately loose, not exact-string
  matching: an LLM's exact phrasing varies between runs even at
  temperature 0 (see this project's own documented refusal-flakiness case),
  so a fact-presence check is what stays meaningful across that variance.
* **`forbidden_facts`** -- none may appear. Catches the class of error this
  project hit repeatedly: a wrong business-rule claim (code produced for a
  role that should have been refused, the wrong program's APIs listed).
* **`must_abstain`** -- the question is genuinely unanswerable from the
  corpus; checks the answer contains the app's refusal phrase and carries
  no citations. Checking the ANSWER TEXT, not just the `abstained` flag,
  catches both ways a request ends up unanswered: the confidence gate
  abstaining before generation ever runs, and the LLM's own grounding-based
  refusal when retrieved context turns out not to support an answer.

```bash
python -m eval.run_generation_eval
```

**Honest limitation:** a single run is a spot check, not a certainty --
the LLM is not perfectly deterministic even at temperature=0. A case that
fails once, alone, is worth a second run before treating it as a real
regression (the script's own output says so). Facts in the shipped set were
chosen to be the parts of the answer that stayed stable across repeated
live runs before being added.

## Layer 3 — The feedback loop (what makes it improve over time)

Both eval sets above only tell you about questions someone already thought
to write down. Real users ask things nobody anticipated -- this project's
own history is full of that (`authenication`, `dahsbaords`, `aisp`, a
region clause collapsing the reranker score). The feedback loop is how a
REAL failure, from REAL usage, turns into a permanent fix and a permanent
test, instead of being seen once and forgotten.

**1. Capture.** Every answer in the chat UI has a 👍/👎, with an optional
comment on 👎 (`app/web/static/app.js`'s `buildFeedbackRow`). Posted to
`/api/feedback` (`app/web/server.py`), logged to `logs/feedback.jsonl`,
keyed by that answer's `request_id` -- the same id already in
`logs/requests.jsonl` for every request, so the two files never need to
know about each other's schema, only share that one key. Every rating is
appended, never overwritten: a user can react, then add a comment later, or
change their mind, and nothing is silently lost.

**2. Triage.** `python -m eval.triage_feedback` (or `--down-only` for just
the negative ratings) joins the two logs and prints each piece of feedback
with its full context in one place: the exact query, role, intent, whether
it abstained, the classifier used, total latency. A bare "👎, wrong API
listed" tells you nothing actionable; with the join, you have everything
needed to reproduce it.

**3. Close the loop.** A real, understood failure becomes a permanent row:
- a retrieval miss -> a new case in `eval/golden_set.json`
- a wrong or hallucinated answer -> a new case in
  `eval/generation_eval_set.json` (`expected_facts`/`forbidden_facts`)
- a classification gap -> a new row in `eval/query_understanding_set.json`
  or `eval/llm_classifier_set.json`

From then on, that exact failure can never silently come back without one
of the eval scripts above catching it first.

## Running everything before shipping a change

There is no single combined "deploy gate" command yet -- each layer is its
own script, run individually (this is what every change in this project's
history has actually done, one script at a time). A natural next step, once
this is used for a while, is wrapping all of them into one pass/fail
command; not built yet because there's no evidence yet of which failure
mode actually recurs most in practice -- exactly the question the feedback
loop above is designed to start answering.

```bash
pytest                                    # unit tests, no DB or network needed
python -m eval.run_eval parallel          # retrieval
python -m eval.run_understanding_eval     # query understanding (+ --retrieval, --llm, --multi)
python -m eval.run_generation_eval        # answer quality
python -m eval.triage_feedback            # review real feedback since the last check
```
