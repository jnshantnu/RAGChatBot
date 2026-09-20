"""Regression tests for retrieval/query_rewrite.py after its word lists moved
into the vocabulary file and its typo/synonym rules became protected-term aware."""
import pytest

from retrieval import query_rewrite as qr


@pytest.fixture(autouse=True)
def corpus_vocab(monkeypatch):
    # The corpus-derived vocabulary normally comes from Postgres; inject a fixed one.
    monkeypatch.setattr(qr, "_domain_vocab", {
        "authentication", "authenticate", "subscriptions", "subscription", "distributor", "distributors",
        "account", "accounts", "model", "models", "opportunity",
    })


def test_word_lists_now_come_from_the_vocabulary_file():
    assert "Partner WebServices" in qr._SYNONYM_PHRASES and "endpoint" in qr._SYNONYM_WORDS
    assert "Co-Term" in qr._BARE_TERMS and "available" in qr._AVAILABILITY_WORDS


@pytest.mark.parametrize("query,expected_retrieval,rules", [
    ("now I want to implement authenication apis...give me sample code in node.js",
     "now I want to implement authentication apis", ["vocabulary_correction", "code_request_stripped_for_retrieval"]),
    ("tell me about subscritpions", "tell me about subscriptions", ["vocabulary_correction"]),
    ("what are my accont details", "what are my account details", ["vocabulary_correction"]),
    ("How do I authenticate to the Partner WebServices API?", "How do I authenticate to the APIs?", ["synonym_normalization"]),
    ("What is the Buy-Sell business model?", "What is the Buy-Sell business model?", []),
])
def test_earlier_behaviour_is_preserved(query, expected_retrieval, rules):
    r = qr.rewrite_query(query, "reseller")
    assert r.rewritten_query == expected_retrieval and r.rules_applied == rules


def test_role_injection_still_fires_on_a_typo_of_available():
    r = qr.rewrite_query("what apis availabe for me", "reseller")
    assert r.rules_applied == ["role_injection"] and r.rewritten_query.startswith("I am a Reseller partner")


def test_bare_term_expansion_still_fires_on_a_typo():
    r = qr.rewrite_query("Co-Trem", "reseller")
    assert r.rules_applied == ["bare_term_expansion"] and "Co-Term" in r.rewritten_query


def test_code_request_split_keeps_the_full_request_for_generation():
    r = qr.rewrite_query("how do I authenticate, give me sample code in node.js", "reseller")
    assert "node.js" not in r.rewritten_query and "node.js" in r.generation_query


def test_typo_correction_does_not_touch_protected_spans():
    text = "see https://x.io/subscritpions and call /v1/distirbutor/list about subscritpions"
    corrected, changed = qr._correct_vocabulary(text)
    assert "https://x.io/subscritpions" in corrected and "/v1/distirbutor/list" in corrected
    assert corrected.endswith("about subscriptions") and changed


def test_synonym_rule_does_not_touch_urls_or_paths_but_still_rewrites_plain_words():
    out, changed = qr._normalize_synonyms("see https://x.io/servces/endpoint and /v1/endpoint/status about servces")
    assert "https://x.io/servces/endpoint" in out and "/v1/endpoint/status" in out
    assert out.endswith("about APIs") and changed


def test_vocab_correction_can_be_skipped():
    r = qr.rewrite_query("tell me about subscritpions", "reseller", vocab_correction=False)
    assert "subscritpions" in r.rewritten_query and r.rules_applied == []
