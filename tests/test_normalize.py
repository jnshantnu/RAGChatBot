import pytest

from retrieval.query_understanding.normalize import (
    clean_whitespace_and_punctuation, correct_typos, normalize_query,
)


def norm(text, vocab, corpus=None):
    return normalize_query(text, vocab, corpus)


# ── the spec's required cases ───────────────────────────────────────────

def test_case1_biz_and_tradein_are_normalized(fixture_vocab):
    r = norm("which APIs should I consume for biz tradein?", fixture_vocab)
    assert r.normalized == "Which APIs should I consume for business trade-in?"
    assert {(t.original, t.normalized, t.reason) for t in r.corrected_terms} == {
        ("biz", "business", "abbreviation"), ("tradein", "trade-in", "synonym"),
    }


def test_case2_biz_and_trade_in_are_normalized_and_implement_is_left_alone(fixture_vocab):
    r = norm("which APIs can I implement for biz trade in?", fixture_vocab)
    assert r.normalized == "Which APIs can I implement for business trade-in?"
    assert "consume" not in r.normalized.lower()  # "implement" is never rewritten into "consume"


def test_case3_publish_api_becomes_expose_api(fixture_vocab):
    assert norm("how do i publish api for order status?", fixture_vocab).normalized == "How do I expose API for order status?"


def test_case4_typo_is_corrected_only_because_the_entity_is_known(fixture_vocab, shipped_vocab):
    known = norm("what auth do I need for the eligiblity API?", fixture_vocab)
    assert known.normalized == "What authentication do I need for the eligibility API?"
    typo = next(t for t in known.corrected_terms if t.reason == "typo")
    assert (typo.original, typo.normalized) == ("eligiblity", "eligibility") and typo.confidence >= 0.85

    # Same sentence against a vocabulary that has no "Eligibility" entity: left exactly as typed.
    unknown = norm("what auth do I need for the eligiblity API?", shipped_vocab)
    assert "eligiblity" in unknown.normalized and not any(t.reason == "typo" for t in unknown.corrected_terms)


def test_case5_error_code_is_untouched(fixture_vocab):
    assert norm("What does error API-4012 mean?", fixture_vocab).normalized == "What does error API-4012 mean?"


def test_case6_endpoint_path_is_untouched_even_when_it_contains_typos_and_abbreviations(fixture_vocab):
    r = norm("Can I use /v1/tradein/eligiblity for biz?", fixture_vocab)
    assert "/v1/tradein/eligiblity" in r.normalized       # neither the synonym nor the typo rule touched the path
    assert r.normalized.endswith("for business?")         # ...but the plain word outside it was still expanded


def test_case7_low_confidence_typo_is_not_silently_corrected(fixture_vocab):
    r = norm("what about elgiblty?", fixture_vocab)
    assert "elgiblty" in r.normalized                       # preserved as typed
    assert not r.corrected_terms
    assert any("elgiblty" in w and "left unchanged" in w for w in r.warnings)


# ── protected terms in general ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "see https://dev.example.com/docs/biz and reply",
    "email biz@dev.example.com about it",
    "run `biz tradein` locally",
    "```\nbiz tradein\n```",
    "uuid 123e4567-e89b-12d3-a456-426614174000 biz-check",
    "version v1.2.3 shipped",
    "SKU-AB12 is a partner id",
])
def test_protected_spans_are_never_modified(fixture_vocab, text):
    r = norm(text, fixture_vocab)
    for span in ["https://dev.example.com/docs/biz", "biz@dev.example.com", "`biz tradein`", "```\nbiz tradein\n```",
                 "123e4567-e89b-12d3-a456-426614174000", "v1.2.3", "SKU-AB12"]:
        if span in text:
            assert span in r.normalized


def test_known_api_names_are_protected(shipped_vocab):
    r = norm("how do I call Get Subscriptions V1", shipped_vocab)
    assert "Get Subscriptions V1" in r.normalized


# ── vocabulary behaviours ───────────────────────────────────────────────

def test_abbreviations_keep_a_leading_capital(fixture_vocab):
    assert norm("Biz rules for prog setup", fixture_vocab).normalized == "Business rules for program setup"


def test_abbreviation_lookalikes_are_left_alone(fixture_vocab):
    text = "Is the device dev-ops config.json internal?"
    assert norm(text, fixture_vocab).normalized == text


def test_synonym_plural_is_preserved(fixture_vocab):
    assert norm("how to invoke APIs", fixture_vocab).normalized == "How to consume APIs"


def test_use_api_key_is_not_rewritten_but_use_api_is(fixture_vocab):
    assert norm("how do I use API key", fixture_vocab).normalized == "How do I use API key"
    assert norm("how do I use API endpoints", fixture_vocab).normalized == "How do I consume API endpoints"


def test_webhook_callback_becomes_webhook(fixture_vocab):
    assert norm("set up a webhook callback", fixture_vocab).normalized == "Set up a webhook"


def test_lowercase_i_becomes_capital_i_but_ie_is_untouched(fixture_vocab):
    assert norm("what can i do, i.e. today", fixture_vocab).normalized == "What can I do, i.e. today"


def test_high_confidence_typo_against_the_corpus_vocabulary(fixture_vocab):
    r = norm("tell me about subscritpions", fixture_vocab, corpus={"subscriptions", "subscription"})
    assert r.normalized == "Tell me about subscriptions"
    assert r.corrected_terms[0].reason == "typo"


def test_plural_and_singular_forms_are_not_flagged_as_typos(fixture_vocab):
    r = norm("describe opportunities", fixture_vocab, corpus={"opportunity"})
    assert not r.warnings and not r.corrected_terms


def test_correct_typos_picks_the_clear_winner_between_singular_and_plural():
    text, corrections, _ = correct_typos("subscritpions", {"subscriptions", "subscription"})
    assert text == "subscriptions" and corrections[0].normalized == "subscriptions"


def test_correct_typos_refuses_an_ambiguous_tie():
    text, corrections, warnings = correct_typos("catalos", {"catalog", "catalon"}, margin=3)
    assert text == "catalos" and not corrections and warnings


def test_short_words_are_never_fuzzy_corrected():
    assert correct_typos("apsi", {"apis", "apps"})[0] == "apsi"


# ── hygiene ─────────────────────────────────────────────────────────────

def test_whitespace_and_punctuation_cleanup():
    assert clean_whitespace_and_punctuation("  what  is   this ??  ") == "what is this?"
    assert clean_whitespace_and_punctuation("a ,b;c") == "a, b; c"
    assert clean_whitespace_and_punctuation("apis...give me") == "apis... give me"


def test_normalization_is_idempotent(fixture_vocab):
    once = norm("which APIs i can implement for biz trade in??", fixture_vocab).normalized
    assert norm(once, fixture_vocab).normalized == once


def test_empty_and_whitespace_only_input(fixture_vocab):
    assert norm("", fixture_vocab).normalized == ""
    assert norm("   ", fixture_vocab).normalized == ""


def test_normalizing_with_an_empty_vocabulary_only_tidies_whitespace(empty_vocab):
    assert norm("  biz   tradein ", empty_vocab).normalized == "Biz tradein"


def test_valid_plurals_of_known_words_are_not_corrected_to_the_singular(fixture_vocab):
    """Regression (found by eval/run_understanding_eval --retrieval): "webhooks" was
    rewritten to "webhook" because only the singular was a candidate."""
    for text in ["Which webhooks can I implement?", "list my subscriptions and quotes", "describe the regions"]:
        r = norm(text, fixture_vocab, corpus={"subscription"})
        assert not any(t.reason == "typo" for t in r.corrected_terms), (text, r.corrected_terms)
    assert norm("Which webhooks can I implement?", fixture_vocab).normalized == "Which webhooks can I implement?"


def test_a_real_typo_of_a_plural_is_still_corrected(fixture_vocab):
    assert norm("which webhoks can I implement?", fixture_vocab).normalized == "Which webhooks can I implement?"


def test_short_valid_words_do_not_produce_typo_warnings(fixture_vocab):
    """"items" scores 80 against "teams" -- noise, not a typo."""
    r = norm("how many line items are allowed", fixture_vocab, corpus={"teams"})
    assert not r.warnings and not r.corrected_terms
