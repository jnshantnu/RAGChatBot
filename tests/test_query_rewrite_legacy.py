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


# ── location-clause stripping (retrieval only; generation keeps the full text) ──

@pytest.mark.parametrize("query,region", [
    ("which APIs i can use if i am based in vietnam", "vietnam"),
    ("which APIs can I use if I'm based in Brazil?", "Brazil"),
    ("which APIs can I use if we are based in Mexico", "Mexico"),
    ("which APIs can I use if we're located in India", "India"),
    ("I am based in Vietnam, which APIs can I use?", "Vietnam"),
])
def test_a_based_in_region_clause_is_stripped_for_retrieval_only(query, region):
    r = qr.rewrite_query(query, "reseller")
    assert region not in r.rewritten_query        # the region word is gone from the SEARCH text...
    assert region in r.generation_query            # ...but the LLM still sees the full request
    assert "location_clause_stripped_for_retrieval" in r.rules_applied


def test_the_stripped_region_still_reaches_generation_verbatim():
    r = qr.rewrite_query("which APIs can I use if I am based in Vietnam", "reseller")
    assert "Vietnam" not in r.rewritten_query and "Vietnam" in r.generation_query


@pytest.mark.parametrize("query", [
    "which APIs can I use in Mexico",              # no "based"/"located" -- not the pattern that collapsed the score
    "what regions does Autodesk operate in",         # "region" as a topic word, not a location clause
    "how do I authenticate to the API",              # no location mention at all
])
def test_unrelated_phrasing_is_left_alone(query):
    r = qr.rewrite_query(query, "reseller")
    assert "location_clause_stripped_for_retrieval" not in r.rules_applied


def test_an_unknown_country_is_not_touched():
    # only vocabulary-listed regions trigger the strip -- an unrecognised place name
    # is not something this rule can safely judge as "a location clause", so it's left alone.
    r = qr.rewrite_query("which APIs can I use if I am based in Atlantis", "reseller")
    assert "location_clause_stripped_for_retrieval" not in r.rules_applied
    assert r.rewritten_query == r.generation_query


def test_both_a_code_request_and_a_location_clause_strip_together():
    query = "give me sample code in Java to call GetOrderStatus if I am based in Vietnam"
    r = qr.rewrite_query(query, "reseller")
    assert set(r.rules_applied) == {"code_request_stripped_for_retrieval", "location_clause_stripped_for_retrieval"}
    assert r.rewritten_query == "to call GetOrderStatus"
    assert r.generation_query == query


def test_stripping_both_clauses_to_nothing_falls_back_to_the_original():
    # once BOTH clauses are removed there is no topic left at all -- searching on an
    # empty string would be worse than searching on the untouched original, so this
    # falls back rather than sending nothing to retrieval.
    query = "give me sample code in Java if I am based in Vietnam"
    r = qr.rewrite_query(query, "reseller")
    assert r.rewritten_query == query and r.rules_applied == []
