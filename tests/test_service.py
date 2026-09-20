import pytest

from retrieval.query_understanding import service
from retrieval.query_understanding.schema import FALLBACK_WARNING, ApiRole, Intent
from retrieval.query_understanding.service import role_phrase_suffix, understand_query


def u(text, vocab, **kw):
    return understand_query(text, "reseller", vocab=vocab, enabled=True, **kw)


# ── the spec's scenarios end to end ─────────────────────────────────────

def test_consume_scenario(fixture_vocab):
    r = u("which APIs should I consume for biz tradein?", fixture_vocab)
    assert r.original_query == "which APIs should I consume for biz tradein?"      # untouched
    assert r.normalized_query == "Which APIs should I consume for business trade-in?"
    assert r.intent is Intent.API_CONSUMPTION and r.api_role is ApiRole.CONSUMES_PLATFORM_API
    assert "consume" in r.retrieval_query.lower()                                    # retrieval query retains consume intent
    assert {"business", "biz", "trade-in", "consume API", "call API"} <= set(r.expansion_terms)
    assert r.program == "Trade-In" and 3 <= len(r.expansion_terms) <= 8


def test_implement_scenario_is_ambiguous_and_never_assumes_consume(fixture_vocab):
    r = u("which APIs can I implement for biz trade in?", fixture_vocab)
    assert r.normalized_query == "Which APIs can I implement for business trade-in?"
    assert r.intent is Intent.API_IMPLEMENTATION and r.api_role is ApiRole.UNCLEAR
    assert r.ambiguity and r.clarifying_question
    assert r.api_role is not ApiRole.CONSUMES_PLATFORM_API


def test_publication_scenario(fixture_vocab):
    r = u("how do i publish api for order status?", fixture_vocab)
    assert r.intent is Intent.API_PUBLICATION and r.api_role is ApiRole.EXPOSES_PARTNER_API


def test_authentication_scenario_corrects_only_the_known_entity(fixture_vocab):
    r = u("what auth do I need for the eligiblity API?", fixture_vocab)
    assert r.intent is Intent.API_AUTHENTICATION
    assert "eligibility" in r.normalized_query and "authentication" in r.normalized_query
    assert [t.reason for t in r.corrected_terms].count("typo") == 1


def test_error_code_and_endpoint_scenarios_keep_identifiers(fixture_vocab):
    assert "API-4012" in u("What does error API-4012 mean?", fixture_vocab).normalized_query
    assert "/v1/trade-in/eligibility" in u("Can I use /v1/trade-in/eligibility?", fixture_vocab).normalized_query


def test_unknown_typo_is_surfaced_as_a_warning_not_corrected(fixture_vocab):
    r = u("what about elgiblty?", fixture_vocab)
    assert "elgiblty" in r.normalized_query and r.warnings and not r.corrected_terms


# ── retrieval query ─────────────────────────────────────────────────────

def test_retrieval_query_defaults_to_the_normalized_query(fixture_vocab):
    r = u("which APIs should I consume for biz tradein?", fixture_vocab)
    assert r.retrieval_query == r.normalized_query and role_phrase_suffix(r) == ""


def test_role_phrase_is_appended_only_when_the_vocabulary_opts_in_and_confidence_is_high(fixture_vocab):
    import dataclasses

    opted_in = dataclasses.replace(fixture_vocab, append_role_phrase=True)
    consume = u("which APIs should I consume for biz tradein?", opted_in)
    assert consume.retrieval_query.endswith("platform APIs that partners consume")
    assert role_phrase_suffix(consume) == "platform APIs that partners consume"

    ambiguous = u("which APIs can I implement for biz trade in?", opted_in)   # role unclear -> no phrase
    assert ambiguous.retrieval_query == ambiguous.normalized_query


def test_expansion_terms_are_bounded(fixture_vocab):
    assert len(u("How do I invoke biz APIs and call prog docs with auth in env config?", fixture_vocab).expansion_terms) <= 8


# ── safety net ──────────────────────────────────────────────────────────

def test_disabled_flag_returns_the_untouched_query(fixture_vocab):
    r = understand_query("which APIs i can implement for biz trade in?", "reseller", vocab=fixture_vocab, enabled=False)
    assert r.is_fallback and r.normalized_query == r.retrieval_query == r.original_query
    assert r.intent is Intent.UNKNOWN and r.api_role is ApiRole.UNCLEAR


def test_unusable_vocabulary_falls_back(empty_vocab):
    r = understand_query("biz question", "reseller", vocab=empty_vocab, enabled=True)
    assert r.is_fallback and r.normalized_query == "biz question" and "vocabulary_unavailable" in r.warnings


def test_any_internal_error_falls_back_instead_of_raising(fixture_vocab, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(service, "classify", boom)
    r = u("which APIs should I consume?", fixture_vocab)
    assert r.is_fallback and FALLBACK_WARNING in r.warnings and r.normalized_query == "which APIs should I consume?"
    assert "falling back" in caplog.text


def test_a_failing_corpus_vocabulary_provider_falls_back(fixture_vocab):
    def broken():
        raise ConnectionError("database down")

    r = u("which APIs should I consume?", fixture_vocab, corpus_vocab=broken)
    assert r.is_fallback


def test_corpus_vocabulary_can_be_a_set_or_a_callable(fixture_vocab):
    for provider in ({"subscriptions"}, lambda: {"subscriptions"}):
        assert "subscriptions" in u("tell me about subscritpions", fixture_vocab, corpus_vocab=provider).normalized_query


def test_timings_out_param_is_filled(fixture_vocab):
    timings = {}
    u("which APIs should I consume?", fixture_vocab, timings=timings)
    assert set(timings) == {"normalization_ms", "classification_ms"}


def test_flag_default_reads_config(monkeypatch, fixture_vocab):
    import config

    monkeypatch.setattr(config, "QUERY_UNDERSTANDING_ENABLED", False)
    assert understand_query("biz", "reseller", vocab=fixture_vocab).is_fallback


def test_result_is_immutable_so_original_query_cannot_drift(fixture_vocab):
    import dataclasses

    r = u("which APIs should I consume?", fixture_vocab)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.original_query = "x"
