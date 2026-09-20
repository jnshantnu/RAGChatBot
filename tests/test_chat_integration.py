"""chat.prepare_answer wiring: query understanding runs before retrieval, is
fail-safe, drives the clarify exit, and reaches generation. Retrieval, the
database and the LLM are all replaced with fakes -- no network, no Postgres."""
from types import SimpleNamespace

import pytest

import chat
from retrieval import query_rewrite as qr
from retrieval.query_understanding import service
from retrieval.query_understanding.schema import QueryUnderstandingResult
from retrieval.rrf import FusedResult


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chunk(i=1):
    return FusedResult(chunk_id=f"c{i}", doc_id="doc", heading="h", chunk_text="text", fused_score=0.1, arms=["keyword"], rerank_score=0.9)


@pytest.fixture
def pipeline(monkeypatch):
    """Patch out Postgres + retrieval; expose what retrieval was asked to search for."""
    calls = []
    state = {"abstain": False}

    def fake_retrieve(conn, query, groups, mode="sequential"):
        calls.append(SimpleNamespace(query=query, groups=groups))
        gate = SimpleNamespace(abstain=state["abstain"], reason="test", best_score=0.5)
        return SimpleNamespace(
            gate=gate, candidates=[_chunk()], top_k=[_chunk()], degraded_rerank=False,
            timings_ms={"retrieve_ms": 1.0}, trace=[],
        )

    monkeypatch.setattr(chat.psycopg, "connect", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(chat, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat, "get_domain_vocab", lambda: {"subscriptions", "subscription", "authentication"})
    monkeypatch.setattr(qr, "_domain_vocab", {"subscriptions", "subscription", "authentication"})
    return SimpleNamespace(calls=calls, state=state)


def test_retrieval_receives_the_normalized_query(pipeline):
    chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme")
    assert pipeline.calls[0].query == "Which APIs should I consume for business trade-in?"


def test_answering_path_returns_a_pending_answer_carrying_the_understanding(pipeline):
    pending = chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme")
    assert isinstance(pending, chat.PendingAnswer)
    assert pending.understanding.intent.value == "api_consumption" and not pending.understanding.is_fallback
    assert pending.query == "which APIs should I consume for biz tradein?"           # original preserved for the answer step


def test_timings_include_the_understanding_stage(pipeline):
    pending = chat.prepare_answer("which APIs should I consume?", "reseller", "acme")
    assert {"understand_ms", "normalization_ms", "classification_ms"} <= set(pending.result.timings_ms)


def test_query_rewrite_trace_records_the_understanding(pipeline):
    pending = chat.prepare_answer("which APIs can I implement for biz trade in?", "reseller", "acme")
    outputs = pending.rewrite_trace.outputs
    assert outputs["understanding"]["intent"] == "api_implementation" and outputs["understanding"]["ambiguity"] is True
    assert "normalization" in outputs["rules_applied"]
    assert outputs["understanding"]["original_query"] == "which APIs can I implement for biz trade in?"


def test_ambiguous_question_with_weak_retrieval_returns_the_clarifying_question(pipeline):
    pipeline.state["abstain"] = True
    r = chat.prepare_answer("which APIs can I implement for biz trade in?", "reseller", "acme")
    assert isinstance(r, chat.ChatResponse)
    assert r.clarification is True and r.abstained is False
    assert "call our platform APIs" in r.answer and r.understanding["ambiguity"] is True


def test_unambiguous_question_with_weak_retrieval_still_abstains(pipeline):
    pipeline.state["abstain"] = True
    r = chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme")
    assert r.abstained is True and r.clarification is False and r.answer == "I don't have that information."


def test_ambiguous_question_with_good_retrieval_goes_to_the_llm_not_the_clarify_exit(pipeline):
    r = chat.prepare_answer("which APIs can I implement for biz trade in?", "reseller", "acme")
    assert isinstance(r, chat.PendingAnswer) and r.understanding.ambiguity


def test_permission_refusal_happens_before_query_understanding(pipeline, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("understanding ran before the permission check")

    monkeypatch.setattr(chat, "understand_query", must_not_run)
    r = chat.prepare_answer("what are the commission payouts?", "reseller", "acme")
    assert r.permission_refused and not pipeline.calls


# ── the fail-safe path ──────────────────────────────────────────────────

def test_disabled_stage_leaves_behaviour_exactly_as_before(pipeline, monkeypatch):
    import config

    monkeypatch.setattr(config, "QUERY_UNDERSTANDING_ENABLED", False)
    pending = chat.prepare_answer("tell me about subscritpions", "reseller", "acme")
    # the legacy rewriter still corrects the typo itself, as it did before this stage existed
    assert pipeline.calls[0].query == "tell me about subscriptions"
    assert pending.understanding.is_fallback and pending.rewrite.rules_applied == ["vocabulary_correction"]


def test_classifier_crash_never_blocks_the_request(pipeline, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(service, "classify", boom)
    pending = chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme")
    assert isinstance(pending, chat.PendingAnswer)
    assert pipeline.calls[0].query == "which APIs should I consume for biz tradein?"      # original text, untouched
    assert pending.understanding.is_fallback and pending.understanding.intent.value == "unknown"
    assert pending.understanding.api_role.value == "unclear"


def test_failure_does_not_trigger_a_clarify_exit(pipeline, monkeypatch):
    monkeypatch.setattr(chat, "understand_query", lambda q, *a, **k: QueryUnderstandingResult.fallback(q, "test"))
    pipeline.state["abstain"] = True
    r = chat.prepare_answer("which APIs can I implement for biz trade in?", "reseller", "acme")
    assert r.abstained is True and r.clarification is False


def test_history_augmentation_still_composes_with_normalization(pipeline):
    history = [{"query": "earlier question", "answer": "earlier answer"}]
    chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme", history=history)
    sent = pipeline.calls[0].query
    assert sent.startswith("earlier question\nearlier answer") and sent.endswith("Which APIs should I consume for business trade-in?")


def test_finish_exposes_the_understanding_and_result_count(pipeline):
    pending = chat.prepare_answer("which APIs should I consume for biz tradein?", "reseller", "acme")
    pending.text, pending.generate_ms, pending.generate_started_at = "An answer. [1]", 5.0, 0.0
    response = pending.finish()
    assert response.understanding["intent"] == "api_consumption" and response.retrieval_result_count == 1
    assert response.rewritten_query == "Which APIs should I consume for business trade-in?"
    assert response.rewrite_rules_applied[0] == "normalization" and response.clarification is False
