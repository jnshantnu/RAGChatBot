import copy
import json

import pytest

from retrieval.query_understanding import vocabulary
from retrieval.query_understanding.vocabulary import DEFAULT_PATH, VocabularyError, parse_vocabulary


def shipped_data():
    with open(DEFAULT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_shipped_vocabulary_file_is_valid(shipped_vocab):
    """Run `pytest` after editing retrieval/query_vocabulary.json -- this is the guard."""
    assert shipped_vocab.abbreviations and shipped_vocab.synonyms and shipped_vocab.action_verbs
    assert shipped_vocab.known_apis and shipped_vocab.known_programs and shipped_vocab.intent_keywords


def test_shipped_vocabulary_seeds_the_specified_entries(shipped_vocab):
    assert shipped_vocab.abbreviations == {
        "biz": "business", "prog": "program", "int": "integration", "dev": "developer", "docs": "documentation",
        "doc": "documentation", "auth": "authentication", "onboard": "onboarding", "env": "environment",
        "config": "configuration",
    }
    pairs = {s.phrase.lower(): s.replacement for s in shipped_vocab.synonyms}
    for phrase, replacement in {
        "trade in": "trade-in", "tradein": "trade-in", "call api": "consume API", "invoke api": "consume API",
        "use api": "consume API", "integrate with api": "consume API", "publish api": "expose API",
        "provide api": "expose API", "webhook callback": "webhook",
    }.items():
        assert pairs[phrase] == replacement
    assert set(shipped_vocab.action_concepts) >= {"consume", "implement", "expose", "authenticate"}


def test_legacy_rewriter_lists_survived_the_move_out_of_code(shipped_vocab):
    legacy = shipped_vocab.legacy
    assert "Partner WebServices" in legacy["synonym_phrases"] and "endpoint" in legacy["synonym_words"]
    assert "Co-Term" in legacy["bare_terms"] and "available" in legacy["availability_words"]
    assert "authenticate" in legacy["morphological_extras"]


def test_abbreviations_do_not_match_inside_hyphenated_or_dotted_tokens(shipped_vocab):
    rx = shipped_vocab.abbreviation_regex
    assert rx.search("dev portal") and rx.search("Biz question")
    assert not rx.search("dev-ops") and not rx.search("config.json") and not rx.search("device") and not rx.search("internal")


def test_empty_object_parses_to_an_empty_vocabulary(empty_vocab):
    assert not empty_vocab.abbreviations and not empty_vocab.synonyms and empty_vocab.abbreviation_regex is None


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.update(abbreviations={"biz": ""}), "abbreviations"),
    (lambda d: d.update(synonyms=[{"phrase": "x"}]), "synonyms[0]"),
    (lambda d: d.update(intent_keywords={"not_an_intent": ["x"]}), "not a known intent"),
    (lambda d: d["retrieval"].update(role_phrases={"sideways": "x"}), "role_phrases"),
    (lambda d: d["typo_correction"].update(warn_threshold=95, high_confidence=85), "warn_threshold"),
    (lambda d: d["typo_correction"].update(min_word_length=1), "min_word_length"),
    (lambda d: d.update(known_programs=[{"aliases": []}]), "known_programs[0]"),
    (lambda d: d.update(protected_terms="OAuth"), "protected_terms"),
])
def test_malformed_vocabulary_is_rejected_with_a_helpful_message(mutate, message):
    data = copy.deepcopy(shipped_data())
    mutate(data)
    with pytest.raises(VocabularyError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse_vocabulary(data)


def test_missing_file_falls_back_to_an_empty_vocabulary_instead_of_crashing(monkeypatch, caplog):
    import config

    monkeypatch.setattr(config, "QUERY_VOCABULARY_PATH", "/nonexistent/vocab.json", raising=False)
    vocabulary.reset_vocabulary_cache()
    try:
        vocab = vocabulary.get_vocabulary()
        assert not vocab.abbreviations
        assert "missing or invalid" in caplog.text
    finally:
        vocabulary.reset_vocabulary_cache()


def test_synonym_not_before_guard(shipped_vocab):
    rule = next(s for s in shipped_vocab.synonyms if s.phrase == "use API")
    assert rule.regex.search("how do I use API endpoints")
    assert not rule.regex.search("how do I use API key")
    assert not rule.regex.search("how do I use API tokens")


def test_candidate_words_come_from_vocabulary_and_entities_not_a_dictionary(fixture_vocab):
    words = fixture_vocab.candidate_words()
    assert {"business", "eligibility", "authentication", "consume"} <= words
    assert "banana" not in words


@pytest.mark.parametrize("word,expected", [
    ("opportunity", {"opportunities"}), ("opportunities", {"opportunity"}),
    ("webhook", {"webhooks"}), ("webhooks", {"webhook"}),
    ("business", {"businesses"}), ("status", {"statuses"}),
    ("account", {"accounts"}), ("subscriptions", {"subscription"}),
])
def test_plural_partners_use_real_english_endings_not_fabricated_words(word, expected):
    assert vocabulary.plural_partners(word, 5) == expected
    assert "opportunitys" not in vocabulary.plural_partners("opportunity", 5)


def test_plural_partners_respect_the_minimum_length():
    assert vocabulary.plural_partners("apis", 5) == set()


@pytest.mark.parametrize("field,value", [
    ("short_word_allowlist", ["api"]),            # 3 letters collide with real words
    ("short_word_allowlist", ["ap1s"]),
    ("short_word_allowlist", "apis"),
    ("transposition_correction", "yes"),
])
def test_bad_typo_settings_are_rejected(field, value):
    data = copy.deepcopy(shipped_data())
    data["typo_correction"][field] = value
    with pytest.raises(VocabularyError, match="typo_correction"):
        parse_vocabulary(data)


def test_shipped_typo_settings(shipped_vocab):
    assert shipped_vocab.typo_transposition_correction is True
    assert set(shipped_vocab.typo_short_word_allowlist) == {"apis", "json"}
