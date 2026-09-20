import dataclasses

import pytest

from retrieval.query_understanding.schema import (
    FALLBACK_WARNING, MAX_EXPANSION_TERMS, ApiRole, CorrectedTerm, Intent, QueryUnderstandingResult,
)


def make(**overrides):
    base = dict(original_query="q", normalized_query="q", retrieval_query="q")
    base.update(overrides)
    return QueryUnderstandingResult(**base)


def test_enum_values_are_exactly_the_specified_ones():
    assert {i.value for i in Intent} == {
        "api_discovery", "api_consumption", "api_implementation", "api_publication", "api_authentication",
        "business_model", "program_policy", "program_eligibility", "onboarding", "troubleshooting", "unknown",
    }
    assert {r.value for r in ApiRole} == {
        "consumes_platform_api", "exposes_partner_api", "bidirectional_integration", "unclear", "not_applicable",
    }


def test_defaults_are_the_safe_unknown_values():
    r = make()
    assert r.intent is Intent.UNKNOWN and r.api_role is ApiRole.UNCLEAR
    assert r.program is None and r.partner_type is None and r.region is None
    assert r.ambiguity is False and r.clarifying_question is None and r.confidence == 0.0


def test_strings_are_coerced_to_enums():
    r = make(intent="api_publication", api_role="exposes_partner_api")
    assert r.intent is Intent.API_PUBLICATION and r.api_role is ApiRole.EXPOSES_PARTNER_API


@pytest.mark.parametrize("field,value", [("intent", "made_up"), ("api_role", "sideways")])
def test_invalid_enum_is_rejected(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 5])
def test_confidence_must_be_between_0_and_1(confidence):
    with pytest.raises(ValueError):
        make(confidence=confidence)
    with pytest.raises(ValueError):
        CorrectedTerm("a", "b", "typo", confidence)


def test_confidence_bounds_are_inclusive():
    assert make(confidence=0.0).confidence == 0.0
    assert make(confidence=1.0).confidence == 1.0


def test_expansion_terms_are_capped():
    r = make(expansion_terms=[f"t{i}" for i in range(20)])
    assert len(r.expansion_terms) == MAX_EXPANSION_TERMS == 8


def test_original_query_cannot_be_modified_after_construction():
    r = make(original_query="which APIs i can implement?")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.original_query = "changed"


def test_fallback_uses_the_original_text_and_guesses_nothing():
    r = QueryUnderstandingResult.fallback("raw text", "why")
    assert r.original_query == r.normalized_query == r.retrieval_query == "raw text"
    assert r.intent is Intent.UNKNOWN and r.api_role is ApiRole.UNCLEAR
    assert r.is_fallback and FALLBACK_WARNING in r.warnings and "why" in r.warnings


def test_to_dict_is_json_safe():
    import json

    r = make(intent="api_consumption", api_role="consumes_platform_api",
             corrected_terms=[CorrectedTerm("biz", "business", "abbreviation", 1.0)], warnings=["w"])
    d = r.to_dict()
    json.dumps(d)  # must not raise
    assert d["intent"] == "api_consumption" and d["api_role"] == "consumes_platform_api"
    assert d["corrected_terms"][0] == {"original": "biz", "normalized": "business", "reason": "abbreviation", "confidence": 1.0}


def test_log_summary_contains_no_query_text():
    r = make(original_query="secret words", normalized_query="secret words", retrieval_query="secret words",
             corrected_terms=[CorrectedTerm("a", "b", "typo", 0.9), CorrectedTerm("c", "d", "abbreviation", 1.0)])
    summary = r.log_summary()
    assert "secret" not in str(summary)
    assert summary["normalization_count"] == 2 and summary["corrected_terms_count"] == 1
