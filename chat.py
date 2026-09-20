"""End-to-end query pipeline: permission refusal -> rewrite -> retrieve -> gate -> generate.

Mirrors the two distinct "no" paths from the case study's architecture:
  - Permission refusal: decided BEFORE retrieval, from intent + role only.
    Never touches the corpus, so the request itself can't be used to probe
    what exists. Names the category, never a record.
  - Abstain: decided AFTER retrieval, from the confidence gate reading the
    post-fusion score spread -- not from permissions.

Query understanding (retrieval/query_understanding/) runs first, right after
the permission check: it normalises the question (abbreviations, canonical
phrases, high-confidence typos -- never touching URLs/paths/codes/IDs) and
classifies its intent and API role with deterministic rules. It is fail-safe:
if it is disabled or errors, the original query flows through unchanged. Its
output feeds the rewriter below, the confidence-gate "clarify" exit, and the
answer prompt.

Query rewriting (retrieval/query_rewrite.py) then runs on the NORMALISED text
(permission refusal already checked the original) and produces
TWO texts, not one -- a retrieval_query and a generation_query, identical
unless the rewriter stripped a trailing format-request clause (e.g. "...give
me sample code in Node.js"). That split exists because the cross-encoder
reranker (retrieval/rerank.py) turned out to be fooled by that clause's own
wording: a page that just points at code samples outranked a page that
actually explains the mechanism, until the clause was left out of the
reranked query specifically (measured, not assumed -- see
_strip_code_request's own comment). Generation still gets the full request,
so it knows to write code; only retrieval searches on the topic alone. Role
is still passed to generate_answer separately too, even when a rewrite
already mentions it -- the rewriter's triggers are pattern-based and won't
catch every phrasing, so the explicit role stays as a general-purpose
backstop rather than the only signal.
"""
import time
from dataclasses import asdict, dataclass, field

import psycopg

import config
from llm.generate import GeneratedAnswer, finalize_answer, generate_answer, stream_answer
from retrieval.pipeline import RetrievalResult, retrieve
from retrieval.query_rewrite import RewriteResult, get_domain_vocab, rewrite_query
from retrieval.query_understanding import QueryUnderstandingResult, role_phrase_suffix, understand_query
from retrieval.trace import TraceStep

# Keyword-triggered intent check for principal-only categories. A real system
# would use a small classifier or the query rewriter; a keyword match is
# enough to demonstrate the "decided before retrieval" shape at POC scale.
PERMISSION_RULES = [
    {
        "keywords": ["commission", "payout", "incentive", "margin"],
        "required_group": "role:principal",
        "category_name": "commission and incentive payout details",
        "restriction_label": "principal-level users",
    },
    {
        "keywords": ["placeorderv2", "place order api", "getmyprice", "get my price"],
        "required_group": "role:distributor",
        "category_name": "Place Order and GetMyPrice API details",
        "restriction_label": "distributor accounts",
    },
]


# Everything the FastAPI endpoint (app/main.py) returns to the client, one
# instance per request. Fields default to the "nothing interesting happened"
# case so early-return paths (refusal, abstain) don't need to set every field.
@dataclass
class ChatResponse:
    query: str
    role: str
    partner: str
    answer: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    permission_refused: bool = False
    degraded_rerank: bool = True
    scores: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    rewritten_query: str | None = None  # set only when query_rewrite.py actually changed something
    rewrite_rules_applied: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)  # debug-mode flowchart data, see retrieval/trace.py
    mode: str = "sequential"  # which retrieval/pipeline.py execution strategy produced this response
    clarification: bool = False  # True when `answer` is a clarifying question rather than an answer or an abstain
    understanding: dict | None = None  # QueryUnderstandingResult.to_dict() -- what the query-understanding stage decided
    retrieval_result_count: int = 0  # candidates that survived rerank (for logging; the UI shows `scores`)


def _user_groups(role: str, partner: str) -> list[str]:
    # Translates a UI role/partner selection into the ACL group list used to
    # filter retrieval (see retrieval/hybrid_search.py's `acl && :groups`
    # predicate). "public" is always included; role adds at most one more group.
    groups = ["public", f"partner:{partner}"]
    if role == "principal":
        groups.append("role:principal")
    elif role == "distributor":
        groups.append("role:distributor")
    return groups


def _visible_scores(results) -> list[dict]:
    """Debug/UI score payload -- everything retrieved is citable now, so
    nothing is filtered out here beyond capping the length for display."""
    return [asdict(r) for r in results][:5]


_HISTORY_RETRIEVAL_ANSWER_CHARS = 600  # cap so one long prior answer doesn't drown out the current question's own terms


def _augment_retrieval_for_history(retrieval_query: str, history: list[dict] | None) -> str:
    # Retrieval only looks at THIS turn's rewritten text by default -- fine
    # for a standalone question, but a follow-up like "as shown in Dashboard
    # idea#1" refers to something the model invented in ITS OWN previous
    # answer (a label, not a corpus term), so a search on the follow-up text
    # alone finds nothing (confirmed: near-zero scores across the board on
    # exactly this case). Folding the last turn's question + answer back in
    # as extra search text resolves it indirectly, without needing real
    # coreference resolution: the prior answer already names the real corpus
    # terms behind "idea #1" (e.g. "Get Subscriptions V1"), so keyword/
    # semantic search can find them again just from text overlap. Only the
    # LAST turn is used, not the whole history -- more would dilute the
    # current question's own terms without helping (the reranker reads at
    # most 256 tokens per chunk anyway, so a sprawling multi-turn retrieval
    # query wouldn't even get fairly compared against it).
    if not history:
        return retrieval_query
    last = history[-1]
    prior_answer = last["answer"][:_HISTORY_RETRIEVAL_ANSWER_CHARS]
    return f"{last['query']}\n{prior_answer}\n\n{retrieval_query}"


def _check_permission_refusal(query: str, user_groups: list[str]) -> str | None:
    # Simple substring match against each rule's keyword list -- if the query
    # mentions a restricted topic AND the session lacks the required group,
    # return a refusal message immediately. Returns None (no rule fired) to
    # let the caller fall through to normal retrieval.
    lowered = query.lower()
    for rule in PERMISSION_RULES:
        if any(kw in lowered for kw in rule["keywords"]) and rule["required_group"] not in user_groups:
            return (
                f"I can't share {rule['category_name']} with this account type -- "
                f"that information is restricted to {rule['restriction_label']}."
            )
    return None


def _rules_applied(understanding: QueryUnderstandingResult | None, rewrite: RewriteResult) -> list[str]:
    """What the UI/trace lists as 'rules applied': the normalisation stage (if it
    changed anything) followed by the rewriter's own rules."""
    applied = ["normalization"] if understanding and understanding.corrected_terms else []
    return applied + list(rewrite.rules_applied)


@dataclass
class PendingAnswer:
    """Retrieval is done and the confidence gate said "answer" -- only the LLM
    call is left. Splitting the request here is what makes streaming possible:
    the caller can stream() the answer as it's written, then finish() to verify
    citations on the complete text and assemble the ChatResponse. answer_query()
    below is the non-streaming wrapper over the same two phases.
    """
    query: str
    role: str
    partner: str
    mode: str
    retrieval_query: str
    generation_query: str
    rewrite: RewriteResult
    rewrite_trace: TraceStep
    result: RetrievalResult
    t_start: float
    history: list[dict] = field(default_factory=list)
    understanding: QueryUnderstandingResult | None = None
    text: str = ""
    generate_ms: float | None = None
    first_token_ms: float | None = None  # since the request started, not since generation started
    generate_started_at: float | None = None

    def stream(self):
        t0 = self.generate_started_at = time.perf_counter()
        for delta in stream_answer(self.generation_query, self.result.top_k, self.role, self.history, self.understanding):
            if self.first_token_ms is None:
                self.first_token_ms = round((time.perf_counter() - self.t_start) * 1000, 1)
            self.text += delta
            yield delta
        self.generate_ms = round((time.perf_counter() - t0) * 1000, 1)

    def complete(self) -> "ChatResponse":
        t0 = self.generate_started_at = time.perf_counter()
        generated = generate_answer(self.generation_query, self.result.top_k, self.role, self.history, self.understanding)
        self.generate_ms = round((time.perf_counter() - t0) * 1000, 1)
        return self.finish(generated)

    def finish(self, generated: GeneratedAnswer | None = None) -> "ChatResponse":
        generated = generated or finalize_answer(self.text, self.result.top_k)
        result = self.result
        result.timings_ms["generate_ms"] = self.generate_ms
        outputs = {
            "answer_chars": len(generated.answer),
            "citations": generated.cited_chunk_ids,
            "uncited_claims_flagged": generated.uncited_claims_flagged,
        }
        if self.first_token_ms is not None:
            result.timings_ms["first_token_ms"] = self.first_token_ms
            outputs["first_token_ms"] = self.first_token_ms
        generate_trace = TraceStep(
            name="Generate Answer",
            inputs={"role": self.role, "chunks_in_context": len(result.top_k)},
            outputs=outputs,
            timing_ms=self.generate_ms, started_at=self.generate_started_at,
            detail={"answer": generated.answer},
        )
        return ChatResponse(
            query=self.query, role=self.role, partner=self.partner,
            answer=generated.answer,
            citations=generated.cited_chunk_ids,
            degraded_rerank=result.degraded_rerank,
            scores=_visible_scores(result.top_k),
            rewritten_query=self.retrieval_query if self.retrieval_query != self.query else None,
            rewrite_rules_applied=_rules_applied(self.understanding, self.rewrite),
            trace=[self.rewrite_trace, *result.trace, generate_trace],
            mode=self.mode,
            understanding=self.understanding.to_dict() if self.understanding else None,
            retrieval_result_count=len(result.candidates),
            timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - self.t_start) * 1000, 1)},
        )


def plan_query(
    query: str, role: str, *, vocab=None, timings: dict | None = None
) -> tuple[QueryUnderstandingResult, RewriteResult, str, str]:
    """Everything that turns the user's raw question into the two texts the
    pipeline actually uses: (understanding, rewrite, retrieval_query,
    generation_query). Shared by prepare_answer() and eval/run_understanding_
    eval.py so the evaluation exercises exactly what production runs.

    Query understanding (retrieval/query_understanding/): normalise the text
    and classify intent / API role. Never raises and never blocks -- on any
    failure (or with QUERY_UNDERSTANDING_ENABLED=false) it hands back the
    original query untouched, flagged as a fallback, and the legacy rewriter
    then does its own typo correction exactly as before.

    Rewrite (retrieval/query_rewrite.py): role injection, synonym
    normalization, bare-term expansion, or a code-request split, whichever
    rules' triggers match -- run on the NORMALISED text. A no-op for queries
    that don't match any trigger. retrieval_query and generation_query differ
    only when a trailing "give me sample code in X" clause got stripped for
    retrieval (see query_rewrite.py's _strip_code_request) -- retrieval
    searches on the topic alone, generation still sees the full request so it
    knows to write code."""
    understanding = understand_query(query, role, vocab=vocab, corpus_vocab=get_domain_vocab, timings=timings)
    rewrite = rewrite_query(understanding.normalized_query, role, vocab_correction=understanding.is_fallback)
    retrieval_query = rewrite.rewritten_query
    role_phrase = role_phrase_suffix(understanding)  # '' unless the vocabulary opts in to appending an API-role phrase
    if role_phrase and role_phrase.lower() not in retrieval_query.lower():
        retrieval_query = f"{retrieval_query} {role_phrase}"
    return understanding, rewrite, retrieval_query, rewrite.generation_query


def prepare_answer(
    query: str, role: str, partner: str, mode: str = "sequential", history: list[dict] | None = None
) -> "ChatResponse | PendingAnswer":
    # Everything up to (not including) the LLM call. Three possible outcomes:
    # permission refusal (no corpus access at all) or confidence-gate abstain
    # (retrieved but not confident enough) each return a finished ChatResponse
    # immediately -- there's nothing to generate; otherwise a PendingAnswer
    # comes back, ready to be streamed or completed.
    # `mode` ("sequential" or "parallel") only picks which execution strategy
    # retrieval/pipeline.py uses -- see that module's docstring for why the
    # retrieval logic itself is identical either way. `history` is prior
    # turns from THIS session only (see app/web/static/app.js -- filtered to
    # the current role/partner before it ever reaches here), each
    # {"query": ..., "answer": ...}; used two different ways below, not one:
    # folded into the retrieval text (_augment_retrieval_for_history) so a
    # follow-up like "idea #1" can still find real chunks, and passed
    # separately to generation so the model can resolve the reference itself
    # -- see llm/generate.py's SYSTEM_PROMPT for why history is never itself
    # a source of facts there.
    t_start = time.perf_counter()
    user_groups = _user_groups(role, partner)

    # Exit 1: refuse before touching the corpus at all, if the query names a
    # restricted category the session's role doesn't have.
    refusal = _check_permission_refusal(query, user_groups)
    if refusal:
        return ChatResponse(
            query=query, role=role, partner=partner,
            answer=refusal, permission_refused=True, mode=mode,
            timings_ms={"total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
        )

    # Query understanding + rewrite -- see plan_query() for what each does and
    # why it can never block the request.
    t0 = time.perf_counter()
    understand_timings: dict = {}
    understanding, rewrite, retrieval_query, generation_query = plan_query(query, role, timings=understand_timings)
    understand_ms = round((time.perf_counter() - t0) * 1000, 1)
    retrieval_query_for_search = _augment_retrieval_for_history(retrieval_query, history)
    rules_applied = _rules_applied(understanding, rewrite)
    rewrite_outputs = {
        "rewritten_query": retrieval_query, "rules_applied": rules_applied,
        # one nested dict -> the flowchart node shows just the key; the full
        # detail is in this step's Outputs/detail card
        "understanding": understanding.to_dict(),
    }
    if generation_query != retrieval_query:
        rewrite_outputs["generation_query"] = generation_query
    if retrieval_query_for_search != retrieval_query:
        rewrite_outputs["retrieval_query_with_history"] = retrieval_query_for_search
    rewrite_trace = TraceStep(
        name="Query Rewrite", inputs={"query": query, "role": role},
        outputs=rewrite_outputs,
        timing_ms=round((time.perf_counter() - t0) * 1000, 1), started_at=t0,
    )

    # The connection is only needed for retrieval -- generation works from the
    # in-memory results, so it no longer sits open during the LLM call.
    with psycopg.connect(config.DATABASE_URL) as conn:
        result = retrieve(conn, retrieval_query_for_search, user_groups, mode=mode)
    result.timings_ms["understand_ms"] = understand_ms
    result.timings_ms.update(understand_timings)  # normalization_ms / classification_ms, for the request log

    def early_response(answer: str, **flags) -> ChatResponse:
        return ChatResponse(
            query=query, role=role, partner=partner, answer=answer,
            degraded_rerank=result.degraded_rerank,
            scores=_visible_scores(result.candidates),
            rewritten_query=retrieval_query if retrieval_query != query else None,
            rewrite_rules_applied=rules_applied,
            trace=[rewrite_trace, *result.trace],
            mode=mode,
            timings_ms={**result.timings_ms, "total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
            understanding=understanding.to_dict(),
            retrieval_result_count=len(result.candidates),
            **flags,
        )

    # Exit 2: retrieval ran, but the confidence gate says nothing found is
    # relevant enough to answer from -- skip the LLM call entirely.
    if result.gate.abstain:
        # If the question was genuinely ambiguous (e.g. "which APIs can I
        # implement" -- call the platform's, or publish one?), the most useful
        # reply is the clarifying question the understanding stage already
        # wrote, not a flat "I don't have that". No LLM call needed for it.
        if understanding.ambiguity and understanding.clarifying_question:
            return early_response(understanding.clarifying_question, clarification=True)
        return early_response("I don't have that information.", abstained=True)

    # Exit 3: a real, cited answer is warranted. result.top_k already includes
    # every guaranteed chunk (e.g. ADSK-BIZ-RULES.md) alongside the
    # competitive top-K -- all of it is citable, see llm/generate.py. The
    # role is passed through explicitly too (see module docstring) so the
    # model can resolve "me" even when the rewriter's triggers didn't fire.
    return PendingAnswer(
        query=query, role=role, partner=partner, mode=mode,
        retrieval_query=retrieval_query, generation_query=generation_query, rewrite=rewrite, rewrite_trace=rewrite_trace,
        result=result, t_start=t_start, history=history or [], understanding=understanding,
    )


def answer_query(
    query: str, role: str, partner: str, mode: str = "sequential", history: list[dict] | None = None
) -> ChatResponse:
    # Non-streaming entry point: same two phases as the streaming UI path, just
    # run to completion. Eval, Compare mode, and any programmatic caller use this.
    prepared = prepare_answer(query, role, partner, mode, history)
    if isinstance(prepared, ChatResponse):
        return prepared
    return prepared.complete()
