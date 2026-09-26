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
  ├─ classify (LLM, optional)  only when the rules say `unknown`; flag-guarded, off by default
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
| `warnings` | e.g. a near-miss typo that was deliberately **not** corrected; `llm_classifier_failed` / `llm_classifier_not_confident` when the optional LLM fallback ran but its answer was not used |
| `classifier` | who decided intent / API role: `rules`, `llm` (the optional fallback), or `none` (stage bypassed) |

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
| tune typo strictness | `typo_correction` (`high_confidence`, `warn_threshold`, `margin`, `warn_min_word_length`, `transposition_correction`) |
| let a short word be the target of a swapped-letter fix | `typo_correction.short_word_allowlist` (4+ letters, e.g. `apis`) |

Matching rules: case-insensitive, whole words, and never inside a hyphenated or
dotted token (`dev` does not match `dev-ops` or `config.json`). Abbreviations
are exact; there is no fuzzy matching on them.

If the file is missing or invalid the app logs an **ERROR**, the stage becomes a
no-op (raw query used), and the legacy rewriter's word lists come back empty.
The service does not crash. `tests/test_vocabulary.py` guards the shipped file.

Settings (env vars, read in `config.py`):
`QUERY_UNDERSTANDING_ENABLED` (default `true`; `false` bypasses the stage) and
`QUERY_VOCABULARY_PATH` (default: the file above). The LLM fallback has its own
settings, described in the next section.

## Optional LLM classifier fallback

Rules only recognise the phrasings the vocabulary lists ("how do I hook my order
system up to your order feed" is `unknown` to them). When -- and only when -- the
rules return `unknown` for a question of 3+ words, **one** call to a small
non-reasoning model classifies it (`llm_classifier.py`).

| env var | default | |
|---|---|---|
| `QUERY_LLM_CLASSIFIER_ENABLED` | `false` | turn the fallback on (`true`), then restart |
| `QUERY_LLM_CLASSIFIER_MODEL` | `openai/gpt-4o-mini` | any OpenRouter chat model; pick a small, non-reasoning one |
| `QUERY_LLM_CLASSIFIER_TIMEOUT_S` | `3` | hard cap on the call; no retries |

Guarantees:

* **Classifies only.** The model returns `{intent, api_role, confidence}`. It never rewrites the search text and never sees document content. The reply is validated against the `Intent` / `ApiRole` enums; anything else is discarded.
* **Never blocks a request.** A timeout, HTTP error, bad JSON, `unknown`, or confidence below 0.6 keeps the rules' result (`classifier: "rules"`, plus a warning saying why).
* **No model-written text reaches the user.** For the "implement, direction unclear" case the clarifying question is still the vocabulary's canonical one.
* **Names sent to the model:** only entities already found in the user's *own* question. The vocabulary's API / program lists are never sent. Access control still comes only from the session's ACL groups.
* The model's confidence is capped at 0.75 so it never outranks a rules-derived classification. The role is made consistent with the intent (a business-model question has no API role, "consumption" implies the platform-call direction, and so on).
* The question text goes to the same provider (OpenRouter) that already receives it for answer generation.

Cost: it runs only on the uncertain minority of questions, and adds one model
round trip (about 0.7-1.6 s measured, capped at the timeout) *before* retrieval
starts on those questions. Watch `llm_classification_ms` in the log.

Judge it with `python -m eval.run_understanding_eval --llm` (13 real calls).
`tests/test_llm_classifier.py` covers everything with a fake model.

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

Three tiers, tried in this order:

1. **Swapped letters** (`transposition_correction`, on by default). A word of 5+ letters that is a rearrangement of **exactly one** known word, reachable by at most 1 adjacent swap (2 for words of 8+ letters), is corrected: `dahsbaords` -> `dashboards`, `cerate` -> `create`. The plain fuzzy ratio charges two edits for a swap, so these scored 80 and 83 and missed the 85 bar. No letter is added, dropped or replaced, so it can only confuse two real words that are anagrams of each other, and it does nothing if two known words are equally close. Recorded with confidence 0.95.
2. **Short-word allowlist** (`short_word_allowlist`, default `apis`, `json`). Words under 5 letters are never fuzzy-matched, but one of these listed 4+ letter words may be reached by a single swap: `aips` -> `apis`, `jsno` -> `json`. Only a swap: `apps`, `apix`, `items` and `teams` are never touched. Entries must be 4+ letters.
3. **Fuzzy ratio**, for one wrong or missing letter in a longer word:

* score >= `high_confidence` (85) and a clear lead over the runner-up -> corrected, recorded in `corrected_terms`;
* score between `warn_threshold` (78) and 85, or an ambiguous tie -> **left as typed**, recorded in `warnings`;
* words shorter than 5 letters are never fuzzy-matched, and near-miss *warnings* are only raised for words of 7+ letters (`warn_min_word_length`) -- shorter words collide too easily (`items` vs `teams`);
* plural/singular pairs are not flagged.

Example: `what dahsbaords i can cerate using aips?` becomes `What dashboards I can create using apis?`
(before the swap tiers it scored 0.0000 on every chunk and abstained). Example: `eligiblity` is corrected to `eligibility` only if an Eligibility
entity is in the vocabulary. In this corpus there isn't one, so it is left alone.

## Location-clause stripping

A "based in <region>" clause is written for the LLM, not for retrieval. Measured
on this corpus: "which APIs can I use" scores 0.32 against the chunk that answers
it (the confidence gate needs >= 0.20); adding "if I am based in Vietnam" -- even
once that chunk was edited to literally contain the word "Vietnam" next to the
relevant API content -- collapsed the same pair to 0.0002 and the gate abstained.
The cross-encoder reads an unrecognized-sounding location as a strong "this
question is about something else" signal, strong enough to outweigh a solid
topical match. This is a bias in the reranker model, not a missing chunk or a
missing fact -- region/partner-type stay soft signals (not used to filter or
re-rank, see "Metadata filters" below), so the search TEXT itself was what hurt.

`retrieval/query_rewrite.py`'s `_strip_location_clause` removes a clause like "if
I am based in <region>" / "I'm located in <region>" from the RETRIEVAL text only,
the same pattern already used for a trailing "give me sample code in X" clause
(see "Retrieval query" below) -- and by the same mechanism: retrieval searches the
topic alone, generation still gets the full request, so the model still knows the
user is in Vietnam and can say so. Only fires on a region from the controlled
vocabulary's `regions` list (never an arbitrary "in <word>"), and both this rule
and the code-request rule are matched independently against the same text and
removed together, so a query with both clauses doesn't lose one because the other
already reduced the leftover text.

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
`understanding_fallback`, `classifier` (`rules`/`llm`/`none`),
`llm_classifier_called`, `llm_classifier_fallback` (called, but the rules'
result was kept), `llm_rewrite_attempted`, `llm_rewrite_used` (the retry
cleared the confidence gate), `retrieval_query_count`, `retrieval_result_count`,
`retrieval_fallback_used`, and in `timings_ms`: `normalization_ms`,
`classification_ms`, `llm_classification_ms`, `llm_rewrite_ms` (each only when that step ran), `understand_ms`, plus the existing retrieve/rerank/generate/
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

## Optional LLM rewrite retry (the "middle path")

Off by default (`QUERY_LLM_REWRITE_ENABLED=true` + restart to enable). This is
the middle ground between the deterministic rewriter (only fixes what's
explicitly in its vocabulary) and always calling an LLM to rewrite every
question (adds latency/cost/non-determinism to every request, including the
majority that already succeed):

**When it runs:** only after the deterministic pipeline's OWN retrieval already
failed the confidence gate, and only when the question isn't the ambiguous
"implement, direction unclear" case (that gets the clarifying question
instead -- a different search wouldn't resolve it). One call, one retry, never
a loop. If it fails, times out, or comes back invalid, the original abstain is
kept untouched -- exactly like every other fallback in this app.

**What it does:** `retrieval/query_understanding/llm_rewrite.py` asks a small
model to clean up the SEARCH text only -- fix typos, drop clauses about the
asker's own circumstances -- never to answer the question. Its output is
validated (non-empty, not wildly longer than the input) before a second
retrieval attempt is made with it. The original question stays authoritative
for generation, exactly like every other rewrite in this app; nothing from
this module ever reaches the answer prompt.

**Measured, not assumed** (`gpt-4o-mini`, real retrieval, real model calls):

| case | outcome |
|---|---|
| `which apis can a distributor use if they just joined this month` | **fixed**: the LLM correctly dropped the irrelevant personal clause (something no fixed vocabulary list could ever anticipate); score 0.20 (abstain) -> 0.99 |
| `which aisp can a distributor use in philipines` | **not fixed**: 0/9 across three prompt variants (including one that explicitly explained letter-jumbling), the model never unscrambled "aisp" -> "apis". A severely scrambled short word turned out to be a genuinely hard task for a small model in one shot -- the deterministic swap-based tier (see "Typo policy") is more reliable for exactly this case, within its own 1-swap safety limit |
| `how do i hook my order system up to your order feed` (a paraphrase, not a typo) | **not fixed**: the model judged it couldn't improve the wording and returned it unchanged |

**Honest takeaway:** this closes a real, different gap than the deterministic
rewriter -- arbitrary personal/circumstantial clauses that no curated
vocabulary could ever list in advance -- but it is not a general typo fixer
and should not be relied on as one; severely scrambled short words and
paraphrase/vocabulary mismatches are still real gaps. Shipped OFF, behind its
own flag, so this can be judged against real production abstains before
anyone turns it on -- the same standard applied to the LLM classifier
fallback and to multi-query retrieval below.

## Tried and rejected: multi-query retrieval

Built, measured, and removed. The idea: also search a second query, the normalized
question plus its `expansion_terms`, fuse the result lists with RRF, and let the
reranker and confidence gate read only the clean question.

What the evaluation (golden set + labeled set, real database and embedding API) showed:

* **No measurable gain.** Golden recall@5 and @10 were 14/14 with and without it (the set is saturated, so it could not show a gain anyway); **0** confidence-gate decisions changed; the only score movements came from the always-included `adsk-biz-rules` chunk entering the competitive list, which changes nothing the LLM sees.
* **Little coverage.** Only 2 of 20 golden and 17 of 32 labeled questions produced an enriched variant at all; most questions have no expansion terms. Many expansion terms are generic (`OAuth`, `access token`, `API key`) and pull in noisy keyword matches.
* **A real regression in plain RRF.** For "How do I use the API key?" the enriched variant pushed the best auth pages out of the reranker's top-20 candidate pool. Protecting all but the last five slots still lost one: RRF rank predicts relevance poorly enough that a chunk ranked 16th can be the cross-encoder's top pick. Only an *additive* pool (the primary query's own top 20 untouched, plus up to 5 variant-only extras) removed the regression.
* **A cost.** One more embedding call (this API's p90 is about 10 s) and 5 more rerank pairs; retrieval wall time rose from a median of about 2.6 s to 4.2 s in a noisy run.

Removed because a disabled, unproven feature in the core retrieval path is code to maintain and
explain for no benefit. **Revisit only with a harder evaluation set** (paraphrased or
misspelled questions the single query actually misses), and keep the two lessons: rerank on the
clean question, and never let extra candidates displace the primary query's own.

## Known limitations

* **Rules first.** Phrasing the rules don't cover is `unknown`/`unclear` unless the optional LLM fallback (above) is switched on. With it on, a model can still be wrong or vary between runs even at temperature 0; on the 13-row fallback set (`eval/llm_classifier_set.json`, rules alone 3/13) it got 8/13 exact, with the misses being a timeout, a low-confidence "stay unknown", a debatable policy-vs-business-model label, and two "implement -> ask which direction" outcomes. Treat that set as a smoke test, not an accuracy claim.
* **The LLM call is sequential.** It runs before retrieval starts, so on the questions that need it the user waits for it. Running it alongside retrieval would hide that latency (not done).
* **The labeled set was written alongside the rules**, so 100% on it overstates real accuracy. Grow it with real traffic, and keep some rows unseen when tuning.
* **Single search query.** `expansion_terms` are computed and reported but not used to widen the search; see "Tried and rejected: multi-query retrieval" below.
* **Normalization does not restructure grammar.** `which APIs i can implement` becomes `Which APIs I can implement`, not `...can I implement`.
* Some vocabulary entries are risky by nature (`int`, `dev`, `env`, `auth`, `scope`, `token`). They are exact whole-word matches, but a real user could mean something else; check the evaluation after editing.
* `auth` expands to `authentication` (not `authorization`) as specified; `scope`/`token` count as authentication cues even in unrelated sentences.
* Region and program extraction only knows the values listed in the vocabulary; this corpus has no region-tagged content.
* The location-clause fix only catches the "if I am/we are based in <region>" phrasing that was measured to collapse the reranker score; a bare "in <region>" with no "based"/"located" wasn't found to have the same problem on this corpus and isn't stripped. Other unrecognized-sounding words could plausibly cause the same collapse -- this is a narrow, evidence-driven fix for the one wording pattern that was actually measured, not a general fix for reranker bias against unfamiliar tokens.
