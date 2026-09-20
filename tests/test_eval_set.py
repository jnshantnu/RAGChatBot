"""The labeled query set (eval/query_understanding_set.json) doubles as a regression
gate: the deterministic stage must keep matching every label. Add a row there
when you add a vocabulary entry or a rule."""
import json
import os

import pytest

from retrieval.query_understanding.service import understand_query

SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "query_understanding_set.json")
CASES = json.load(open(SET_PATH, encoding="utf-8"))


def test_the_set_is_big_enough_to_mean_something():
    assert len(CASES) >= 30


@pytest.mark.parametrize("case", CASES, ids=[c["query"][:50] for c in CASES])
def test_each_labeled_query_matches(case, shipped_vocab):
    r = understand_query(case["query"], case.get("role", "reseller"), vocab=shipped_vocab, enabled=True)
    assert (r.intent.value, r.api_role.value, r.ambiguity) == (case["intent"], case["api_role"], case["ambiguity"])
    assert r.original_query == case["query"]
    for fragment in case.get("normalized_contains", []):
        assert fragment.lower() in r.normalized_query.lower(), f"{fragment!r} missing from {r.normalized_query!r}"
    assert 0.0 <= r.confidence <= 1.0 and len(r.expansion_terms) <= 8


def test_llm_classifier_eval_rows_all_fall_through_the_rules(shipped_vocab):
    """The LLM eval set is only meaningful if the rules return `unknown` for every row --
    otherwise the fallback is never exercised. Reword a row that starts being handled by the rules."""
    import json, os
    from retrieval.query_understanding import understand_query

    path = os.path.join(os.path.dirname(__file__), "..", "eval", "llm_classifier_set.json")
    for case in json.load(open(path, encoding="utf-8")):
        r = understand_query(case["query"], "reseller", vocab=shipped_vocab, enabled=True, llm_enabled=False)
        assert r.intent.value == "unknown", case["query"]
