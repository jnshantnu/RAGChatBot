# Query understanding

A stage that runs **before retrieval**. It cleans up the user's question
(abbreviations, canonical phrases, obvious typos), works out what kind of
question it is (especially *which direction an API points*), and hands that to
retrieval and to the answer prompt. It never blocks a request: if it is off or
fails, the original question flows through exactly as it did before this stage
existed.

Code: `retrieval/query_understanding/` · Vocabulary: `retrieval/query_vocabulary.json`
· Tests: `tests/` · Evaluation: `eval/run_understanding_eval.py`

## Request flow

```
user question
  │  (permission refusal check -- on the RAW question, before anything below)
  ▼
query understanding            retrieval/query_understanding/
  ├─ mask protected spans      URLs, paths, code, error codes, IDs, versions, API names...
  ├─ normalize                 whitespace -> abbreviations -> synonyms -> high-confidence typos
  ├─ classify (rules)          intent, API role, ambiguity, program / partner type / region
  └─ QueryUnderstandingResult  (see schema below)
  ▼
query rewrite                  retrieval/query_rewrite.py   -- runs on the NORMALIZED text
  ├─ retrieval_query           topic-focused (a trailing "give me code in X" clause is dropped)
  └─ generation_query          the full request
  ▼
retrieval (unchanged)          keyword + semantic -> RRF -> cross-encoder rerank -> confidence gate
  ├─ gate abstains AND question is ambiguous  -> return the clarifying question (no LLM call)
  ├─ gate abstains                            -> "I don't have that information."
  ▼
answer generation              llm/generate.py -- prompt also receives the understanding
  ▼
citations / response           the response carries `understanding`, `clarification`
```

Orchestration lives in `chat.plan_query()` / `chat.prepare_answer()`.

## Result schema (`QueryUnderstandingResult`)

A frozen dataclass, validated on construction (bad enum or a confidence outside
0..1 raises; `expansion_terms` is capped at 8).

| field | meaning |
|---|---|
| `original_query` | exactly what the user typed. Never modified (the object is frozen). |
| `normalized_query` | abbreviations expanded, canonical phrases applied, high-confidence typos fixed. Same meaning, same word order. |
| `retrieval_query` | canonical search text. Equals `normalized_query` unless the vocabulary opts in to appending an API-role phrase (see "Retrieval query"). |
| `intent` | `api_discovery` `api_consumption` `api_implementation` `api_publication` `api_authentication` `business_model` `program_policy` `program_eligibility` `onboarding` `troubleshooting` `unknown` |
| `api_role` | `consumes_platform_api` `exposes_partner_api` `bidirectional_integration` `unclear` `not_applicable` |
| `program` / `partner_type` / `region` | only ever a value that is in the text (or, for `partner_type`, the signed-in session's role). Otherwise `null`. |
| `entities` | known API / program names found in the text |
| `expansion_terms` | 3-8 extra search terms (concept phrases plus the user's own alias, e.g. `business` and `biz`) |
| `corrected_terms` | audit trail: `{original, normalized, reason: abbreviation\|synonym\|typo, confidence}` |
| `ambiguity` / `clarifying_question` | true when the wording could mean more than one API direction |
| `confidence` | 0..1 for the classification |
| `warnings` | e.g. a near-miss typo that was deliberately **not** corrected |

`QueryUnderstandingResult.fallback(text)` is the "do nothing" result: the text
is used as-is, `intent=unknown`, `api_role=unclear`, and `is_fallback` is true.

### How "implement" is handled

The classifier never treats *implement* as *consume*:

| the user says | intent | api_role | ambiguity |
|---|---|---|---|
| "which APIs should I **consume**..." / call / invoke / integrate with | `api_consumption` | `consumes_platform_api` | no |
| "how do I **publish** / expose / provide an API..." | `api_publication` | `exposes_partner_api` | no |
| "which APIs can I **implement**..." (no direction given) | `api_implementation` | `unclear` | **yes** + clarifying question |
| "build an integration that **calls** X and **publishes** a webhook" | `api_implementation` | `bidirectional_integration` | no |
| "which APIs are available to me" | `api_discovery` | `unclear` | no |

## Editing the controlled vocabulary (no code changes)

Everything a product/program owner is likely to change is in
**`retrieval/query_vocabulary.json`**. Edit it, then **restart the app service**
(it is loaded once at startup). Then run `pytest` -- it validates the file.

| to... | edit |
|---|---|
| add an abbreviation (`biz` -> `business`) | `abbreviations` |
| add a canonical phrase (`trade in` -> `trade-in`) | `synonyms` -- optional `not_before` lists words that block the match (e.g. `use API` is not rewritten in `use API key`) |
| teach it a new API or program | `known_apis` / `known_programs` (with `aliases`) -- also makes typo correction and entity extraction aware of it |
| add a term that must never be touched | `protected_terms` |
| change how a direction is recognised | `action_verbs` (`consume`, `expose`, `implement`, `weak_consume`, `authenticate`) |
| add keywords for a non-API intent | `intent_keywords` (`stem*` = prefix match) |
| change the clarifying question | `clarifying_questions.api_direction` |
| add a region / partner type | `regions` / `partner_types` |
| tune typo strictness | `typo_correction` (`high_confidence`, `warn_threshold`, `margin`, `warn_min_word_length`) |

Matching rules: case-insensitive, whole words, and never inside a hyphenated or
dotted token (`dev` does not match `dev-ops` or `config.json`). Abbreviations
are exact; there is no fuzzy matching on them.

If the file is missing or invalid the app logs an **ERROR**, the stage becomes a
no-op (raw query used), and the legacy rewriter's word lists come back empty.
The service does not crash. `tests/test_vocabulary.py` guards the shipped file.

Settings (env vars, read in `config.py`):
`QUERY_UNDERSTANDING_ENABLED` (default `true`; `false` bypasses the stage) and
`QUERY_VOCABULARY_PATH` (default: the file above).

## Protected terms

Spans that are **never** rewritten, expanded or spell-corrected:
URLs, e-mail addresses, fenced and inline code, endpoint paths (`/v1/...`,
`v2/oauth/token`), error codes (`API-4012`), UUIDs, version strings (`v2.1.0`),
uppercase HTTP methods, identifiers mixing letters and digits (SKUs, partner and
program codes), CamelCase API names (`GetSubscriptions`), plus everything in
`protected_terms` and `known_apis`.

Mechanism (`protected.py`): each span is swapped for an opaque sentinel that
contains no letters, the rules run, then the exact original text is restored.
The legacy rewriter's typo and "service -> APIs" rules use the same masking (the
"service" rule shields only *structural* spans, since it exists to rewrite
product-name words like `WebServices`).

## Typo policy

Corrections go **only** to words in a controlled candidate set: the
vocabulary's own words, known API/program names, and (in the app) the
corpus-derived vocabulary. Never a generic dictionary -- that is how a
spellchecker corrupts API names.

* score >= `high_confidence` (85) and a clear lead over the runner-up -> corrected, recorded in `corrected_terms`;
* score between `warn_threshold` (78) and 85, or an ambiguous tie -> **left as typed**, recorded in `warnings`;
* words shorter than 5 letters are never fuzzy-matched, and near-miss *warnings* are only raised for words of 7+ letters (`warn_min_word_length`) -- shorter words collide too easily (`items` vs `teams`);
* plural/singular pairs are not flagged.

Example: `eligiblity` is corrected to `eligibility` only if an Eligibility
entity is in the vocabulary. In this corpus there isn't one, so it is left alone.

## Retrieval query

By default the search text is the normalized question. The vocabulary can opt in
(`retrieval.append_role_phrase: true`) to appending a short phrase naming the API
direction when the role is confidently known (`retrieval.role_phrases`). It is
**off**: the cross-encoder reranker is very sensitive to wording, so turn it on
only after `python -m eval.run_understanding_eval --retrieval` shows the
`phrase` plan doesn't move the confidence gate.

## Answer prompt

`llm/generate.py` adds a "Query understanding" block (original question -- marked
authoritative --, normalized wording, intent, API role, ambiguity, clarifying
question) and prompt rules: rewrites are retrieval aids only; never turn
"consume" into "implement"/"expose"; state the assumed interpretation
("Assuming you mean APIs your application calls from our platform..." /
"...an API or webhook your system exposes..."); if ambiguous and the sources
don't settle it, ask the clarifying question before recommending an API. The
grounding, citation and exact-refusal rules are unchanged.

## Observability

One JSON line per request in `logs/requests.jsonl`. New fields: `request_id`
(also sent to the browser in the first SSE event), `intent`, `api_role`,
`ambiguity`, `clarification`, `normalization_count`, `corrected_terms_count`,
`understanding_fallback`, `retrieval_query_count`, `retrieval_result_count`,
`retrieval_fallback_used`, and in `timings_ms`: `normalization_ms`,
`classification_ms`, `understand_ms`, plus the existing retrieve/rerank/generate/
total timings. These are counts and categories only; the log already recorded
`query` before this stage existed and nothing else adds user text. The debug
view's "Query Rewrite" step shows the full result.

## How to add a new intent

1. Add the value to `Intent` in `schema.py`.
2. Add its keywords under `intent_keywords` in the vocabulary (the loader rejects names that aren't an `Intent`).
3. If it needs a precedence decision, add a branch in `rules.py:classify()` (order matters: first match wins).
4. Add rows to `eval/query_understanding_set.json` and run `pytest`.

## Metadata filters

The chunk index has no program / region / API-role fields today, only ACL, so
extracted values (`program`, `region`, `partner_type`) are **soft signals**:
they appear in the result, the debug trace and the answer prompt, and can be
folded into the search text, but nothing is excluded by them. Access control
still comes only from the signed-in session's ACL groups -- never from anything
this stage extracts. Hard filters need documents tagged at ingest time.

## Running tests and the evaluation

```bash
pip install -r requirements-dev.txt     # pytest (dev only; the app doesn't need it)
pytest                                  # unit + integration tests, no database or network
python -m eval.run_understanding_eval                # classification accuracy on the labeled set
python -m eval.run_understanding_eval --retrieval    # + retrieval comparison (needs Postgres + embedding API)
python -m eval.run_eval parallel                     # existing golden-set retrieval eval
```

Recommended evaluation examples (`eval/query_understanding_set.json`, 32 rows):
the spec's trade-in / eligibility examples (fictional entities -- tests use
fixture vocabularies for those), plus real-corpus questions: `authenication`
typos, "which APIs can I use", `Get Account` / `Get Subscriptions V1`
consumption, webhook publication, Buy-Sell vs NxM, troubleshooting phrasing.

## Known limitations

* **Rules, not a model.** Phrasing the rules don't cover falls back to `unknown`/`unclear`. A small LLM classifier for the uncertain cases is a planned, separate, flag-guarded step.
* **The labeled set was written alongside the rules**, so 100% on it overstates real accuracy. Grow it with real traffic, and keep some rows unseen when tuning.
* **Single search query today.** Searching with several query forms (original + enriched) and merging is a planned, separate step; `expansion_terms` are computed and reported but not yet used to widen the search.
* **Normalization does not restructure grammar.** `which APIs i can implement` becomes `Which APIs I can implement`, not `...can I implement`.
* Some vocabulary entries are risky by nature (`int`, `dev`, `env`, `auth`, `scope`, `token`). They are exact whole-word matches, but a real user could mean something else; check the evaluation after editing.
* `auth` expands to `authentication` (not `authorization`) as specified; `scope`/`token` count as authentication cues even in unrelated sentences.
* Region and program extraction only knows the values listed in the vocabulary; this corpus has no region-tagged content.
