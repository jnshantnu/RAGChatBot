"""The optional LLM classifier (plan item 9). No network: the model is a fake."""
import json
from types import SimpleNamespace

import pytest

from retrieval.query_understanding import understand_query
from retrieval.query_understanding.llm_classifier import (
    LlmClassifierError, LlmVerdict, build_messages, classify_with_llm, merge_verdict, needs_llm, parse_verdict,
)
from retrieval.query_understanding.rules import classify
from retrieval.query_understanding.schema import ApiRole, Intent

UNKNOWN_Q = "how do I hook my order system up to your order feed"   # nothing in the rules matches this phrasing


class FakeClient:
    """Stands in for the OpenAI SDK client."""
    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def reply(intent, role, confidence=0.8):
    return json.dumps({"intent": intent, "api_role": role, "confidence": confidence})


def verdict_fn(intent, role, confidence=0.8, calls=None):
    def fn(question, entities):
        if calls is not None:
            calls.append((question, entities))
        return LlmVerdict(Intent(intent), ApiRole(role), confidence)
    return fn


def failing(exc):
    def fn(question, entities):
        raise exc
    return fn


# ---------- when is the model asked ----------

def test_the_rules_alone_cannot_classify_the_probe_question(shipped_vocab):
    c = classify(UNKNOWN_Q, shipped_vocab)
    assert c.intent is Intent.UNKNOWN


def test_flag_off_by_default_never_calls_the_model(shipped_vocab):
    calls = []
    r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=True, llm=verdict_fn("api_consumption", "consumes_platform_api", calls=calls))
    assert calls == [] and r.classifier == "rules" and r.intent is Intent.UNKNOWN


def test_model_is_asked_only_when_rules_return_unknown(shipped_vocab):
    calls = []
    llm = verdict_fn("program_policy", "not_applicable", calls=calls)
    for q in ("which APIs should I consume for orders?", "What does error API-4012 mean?", "which APIs can I implement?"):
        r = understand_query(q, "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=True, llm=llm)
        assert r.classifier == "rules"
    assert calls == []


@pytest.mark.parametrize("q", ["thanks", "hello there", "ok"])
def test_trivially_short_questions_are_not_sent_to_the_model(shipped_vocab, q):
    c = classify(q, shipped_vocab)
    assert not needs_llm(c, q)


def test_model_result_replaces_unknown_and_is_marked(shipped_vocab):
    timings, calls = {}, []
    r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=True, timings=timings,
                         llm=verdict_fn("api_consumption", "consumes_platform_api", 0.9, calls))
    assert (r.intent, r.api_role, r.classifier) == (Intent.API_CONSUMPTION, ApiRole.CONSUMES_PLATFORM_API, "llm")
    assert r.ambiguity is False and r.confidence == 0.75          # capped: the rules' confidences outrank the model's
    assert "llm_classification_ms" in timings and len(calls) == 1
    assert r.original_query == UNKNOWN_Q and r.retrieval_query == r.normalized_query == "How do I hook my order system up to your order feed"  # search text untouched by the model


def test_llm_implementation_without_direction_asks_the_vocabulary_question(shipped_vocab):
    r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=True,
                         llm=verdict_fn("api_implementation", "unclear", 0.7))
    assert r.ambiguity is True
    assert r.clarifying_question == shipped_vocab.clarifying_questions["api_direction"]   # canonical text, never model-written


# ---------- failure handling: the rules' result always survives ----------

@pytest.mark.parametrize("exc", [LlmClassifierError("timeout"), TimeoutError("slow"), RuntimeError("boom"), ValueError("bad")])
def test_any_failure_keeps_the_rules_result(shipped_vocab, exc):
    timings = {}
    r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=True, timings=timings, llm=failing(exc))
    assert (r.intent, r.api_role, r.classifier) == (Intent.UNKNOWN, ApiRole.UNCLEAR, "rules")
    assert "llm_classifier_failed" in r.warnings and not r.is_fallback
    assert "llm_classification_ms" in timings           # the attempt is still timed


def test_unknown_or_low_confidence_answers_are_ignored(shipped_vocab):
    for verdict in (verdict_fn("unknown", "unclear", 0.9), verdict_fn("api_consumption", "consumes_platform_api", 0.4)):
        r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=True, llm=verdict)
        assert r.intent is Intent.UNKNOWN and r.classifier == "rules" and "llm_classifier_not_confident" in r.warnings


def test_stage_off_bypasses_the_model_too(shipped_vocab):
    calls = []
    r = understand_query(UNKNOWN_Q, "reseller", vocab=shipped_vocab, enabled=False, llm_enabled=True, llm=verdict_fn("onboarding", "not_applicable", calls=calls))
    assert calls == [] and r.is_fallback and r.classifier == "none"


# ---------- merge_verdict keeps the model's answer internally consistent ----------

@pytest.mark.parametrize("intent,role_in,role_out", [
    ("business_model", "consumes_platform_api", "not_applicable"),
    ("troubleshooting", "unclear", "not_applicable"),
    ("api_consumption", "unclear", "consumes_platform_api"),
    ("api_publication", "consumes_platform_api", "exposes_partner_api"),
    ("api_discovery", "consumes_platform_api", "unclear"),
    ("api_authentication", "not_applicable", "unclear"),
])
def test_merge_coerces_role_to_match_intent(shipped_vocab, intent, role_in, role_out):
    base = classify(UNKNOWN_Q, shipped_vocab)
    merged = merge_verdict(base, LlmVerdict(Intent(intent), ApiRole(role_in), 0.8), shipped_vocab)
    assert merged.api_role.value == role_out


# ---------- the model call itself ----------

def test_call_sends_the_question_and_only_the_names_the_user_typed(fixture_vocab):
    client = FakeClient(reply("api_consumption", "consumes_platform_api"))
    verdict = classify_with_llm("how do I call Order Status from my app", ("Order Status",), client=client, model="m", timeout=2.5)
    assert verdict.intent is Intent.API_CONSUMPTION
    call = client.calls[0]
    assert call["model"] == "m" and call["timeout"] == 2.5 and call["temperature"] == 0
    text = json.dumps(call["messages"])
    assert "Order Status" in text and "Trade-In Eligibility" not in text   # the vocabulary's other names are never sent


def test_prompt_contains_no_entity_line_when_the_user_named_nothing():
    user = build_messages("how do I do that", ())[1]["content"]
    assert user == "Question: how do I do that"


@pytest.mark.parametrize("content", [
    "", "not json", "{}", '{"intent": "made_up", "api_role": "unclear", "confidence": 0.9}',
    '{"intent": "onboarding", "api_role": "sideways", "confidence": 0.9}',
    '{"intent": "onboarding", "api_role": "not_applicable", "confidence": 3}',
    '{"intent": "onboarding", "api_role": "not_applicable", "confidence": "high"}',
])
def test_off_contract_replies_are_errors(content):
    with pytest.raises(LlmClassifierError):
        classify_with_llm("a question that is long enough", client=FakeClient(content), model="m", timeout=1)


def test_reply_wrapped_in_a_code_fence_is_still_parsed():
    v = parse_verdict("```json\n" + reply("onboarding", "not_applicable", 0.7) + "\n```")
    assert v.intent is Intent.ONBOARDING


def test_transport_errors_become_classifier_errors():
    with pytest.raises(LlmClassifierError, match="TimeoutError"):
        classify_with_llm("a question that is long enough", client=FakeClient(error=TimeoutError("read timed out")), model="m", timeout=1)


def test_missing_api_key_is_an_error_not_a_crash(monkeypatch):
    import config
    from retrieval.query_understanding import llm_classifier

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm_classifier, "_default_client", None)
    with pytest.raises(LlmClassifierError, match="OPENROUTER_API_KEY"):
        classify_with_llm("a question that is long enough")
