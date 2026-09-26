"""chat.py's wiring of the optional LLM rewrite retry: only tried after an
abstain, never for the ambiguous/clarify case, never more than once, and any
failure keeps the original abstain untouched. Retrieval, Postgres and the
LLM rewrite call are all replaced with fakes -- no network, no database."""
from types import SimpleNamespace

import pytest

import chat
from retrieval import query_rewrite as qr
from retrieval.query_understanding import llm_rewrite


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chunk():
    from retrieval.rrf import FusedResult
    return FusedResult(chunk_id="c1", doc_id="doc", heading="h", chunk_text="text", fused_score=0.1, arms=["keyword"], rerank_score=0.9)


@pytest.fixture
def pipeline(monkeypatch):
    """abstain_for maps a query TEXT to its gate outcome; unlisted queries abstain.
    Every retrieve() call is recorded so tests can check how many happened and with what text."""
    calls = []
    state = {"abstain_for": {}}

    def fake_retrieve(conn, query, groups, mode="sequential"):
        calls.append(query)
        abstain = state["abstain_for"].get(query, True)
        gate = SimpleNamespace(abstain=abstain, reason="weak_rerank_score", best_score=0.9 if not abstain else 0.001)
        return SimpleNamespace(
            gate=gate, candidates=[_chunk()], top_k=[_chunk()], degraded_rerank=False,
            timings_ms={"retrieve_ms": 1.0}, trace=[],
        )

    monkeypatch.setattr(chat.psycopg, "connect", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(chat, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat, "get_domain_vocab", lambda: {"subscriptions", "subscription", "authentication"})
    monkeypatch.setattr(qr, "_domain_vocab", {"subscriptions", "subscription", "authentication"})
    return SimpleNamespace(calls=calls, state=state)


QUERY = "which apis can a distributor use in philipines"


def enable_flag(monkeypatch, enabled=True):
    import config
    monkeypatch.setattr(config, "QUERY_LLM_REWRITE_ENABLED", enabled)


# ---------- when it is (and isn't) tried ----------

def test_flag_off_never_calls_the_llm_rewrite(pipeline, monkeypatch):
    enable_flag(monkeypatch, False)
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert r.abstained is True and r.llm_rewrite_attempted is False and len(pipeline.calls) == 1


def test_flag_on_but_first_attempt_succeeds_never_calls_the_llm_rewrite(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    pipeline.state["abstain_for"] = {"Which apis can a distributor use in philipines": False}
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert isinstance(r, chat.PendingAnswer) and r.llm_rewrite_attempted is False


def test_ambiguous_clarify_case_is_never_given_to_the_llm_rewrite(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    called = []
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: called.append(q) or "irrelevant")
    r = chat.prepare_answer("which APIs can I implement for biz trade in?", "reseller", "acme")
    assert r.clarification is True and called == [] and len(pipeline.calls) == 1


# ---------- the retry itself ----------

def test_a_successful_rewrite_and_retry_replaces_the_result(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    pipeline.state["abstain_for"] = {"which apis can a distributor use": False}
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: "which apis can a distributor use")
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert isinstance(r, chat.PendingAnswer)
    assert pipeline.calls == ["Which apis can a distributor use in philipines", "which apis can a distributor use"]
    assert r.llm_rewrite_attempted is True and r.llm_rewrite_used is True

    final = r.finish()
    assert final.llm_rewrite_attempted is True and final.llm_rewrite_used is True
    assert any(step.name == "LLM Query Rewrite Retry" for step in final.trace)
    step = next(s for s in final.trace if s.name == "LLM Query Rewrite Retry")
    assert step.outputs["rewritten_query"] == "which apis can a distributor use" and step.outputs["used"] is True


def test_a_retry_that_still_abstains_keeps_the_final_abstain(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    # both the original and the rewritten text fail -- realistic "genuinely not in the docs" case
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: "a cleaned up but still unanswerable query")
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert r.abstained is True and len(pipeline.calls) == 2
    assert r.llm_rewrite_attempted is True and r.llm_rewrite_used is False
    step = next(s for s in r.trace if s.name == "LLM Query Rewrite Retry")
    assert step.outputs["used"] is False


def test_a_model_failure_keeps_the_original_abstain_with_one_retrieve_call(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: (_ for _ in ()).throw(llm_rewrite.LlmRewriteError("timed out")))
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert r.abstained is True and len(pipeline.calls) == 1     # no wasted second retrieval call
    assert r.llm_rewrite_attempted is True and r.llm_rewrite_used is False
    step = next(s for s in r.trace if s.name == "LLM Query Rewrite Retry")
    assert step.outputs["error"] is not None and step.outputs["rewritten_query"] is None


def test_an_unchanged_rewrite_does_not_waste_a_second_retrieve_call(pipeline, monkeypatch):
    enable_flag(monkeypatch)
    normalized = "Which apis can a distributor use in philipines"
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: q)  # "I couldn't improve it"
    r = chat.prepare_answer(QUERY, "distributor", "acme")
    assert pipeline.calls == [normalized]      # only the one call -- retrying with identical text would be pointless
    assert r.abstained is True and r.llm_rewrite_attempted is True and r.llm_rewrite_used is False


def test_it_never_loops_more_than_one_retry(pipeline, monkeypatch):
    calls = []
    enable_flag(monkeypatch)
    monkeypatch.setattr(llm_rewrite, "rewrite_with_llm", lambda q, **k: calls.append(q) or "still no good")
    chat.prepare_answer(QUERY, "distributor", "acme")
    assert len(calls) == 1     # the rewrite call itself only ever happens once, however the retry turns out
