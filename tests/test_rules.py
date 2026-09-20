import pytest

from retrieval.query_understanding.rules import classify
from retrieval.query_understanding.schema import ApiRole, Intent


def c(text, vocab, role=None, extra=()):
    return classify(text, vocab, role, extra)


def test_consume_is_api_consumption_and_consumes_platform_api(fixture_vocab):
    r = c("Which APIs should I consume for business trade-in?", fixture_vocab, extra=["business", "biz", "trade-in"])
    assert r.intent is Intent.API_CONSUMPTION and r.api_role is ApiRole.CONSUMES_PLATFORM_API
    assert not r.ambiguity and r.clarifying_question is None
    assert {"consume API", "call API", "business", "biz", "trade-in"} <= set(r.expansion_terms)


def test_implement_is_api_implementation_and_is_ambiguous_not_assumed_to_be_consume(fixture_vocab):
    r = c("Which APIs can I implement for business trade-in?", fixture_vocab)
    assert r.intent is Intent.API_IMPLEMENTATION
    assert r.api_role is ApiRole.UNCLEAR                     # NOT consumes_platform_api
    assert r.ambiguity is True
    assert "call our platform APIs" in r.clarifying_question and "publish an API/webhook" in r.clarifying_question


def test_publish_is_api_publication_and_exposes_partner_api(fixture_vocab):
    r = c("How do I expose API for order status?", fixture_vocab)
    assert r.intent is Intent.API_PUBLICATION and r.api_role is ApiRole.EXPOSES_PARTNER_API and not r.ambiguity


def test_authentication_question(fixture_vocab):
    r = c("What authentication do I need for the eligibility API?", fixture_vocab)
    assert r.intent is Intent.API_AUTHENTICATION and not r.ambiguity
    assert {"OAuth", "client credentials", "access token"} <= set(r.expansion_terms)


def test_oauth_alone_is_recognised_even_though_it_is_a_protected_term(shipped_vocab):
    assert c("How does OAuth work here?", shipped_vocab).intent is Intent.API_AUTHENTICATION


def test_implement_with_an_explicit_direction_is_resolved(shipped_vocab):
    consume = c("How do I build an integration that calls the Get Account API?", shipped_vocab)
    assert consume.intent is Intent.API_IMPLEMENTATION or consume.intent is Intent.API_CONSUMPTION
    assert consume.api_role is ApiRole.CONSUMES_PLATFORM_API and not consume.ambiguity

    expose = c("How do I implement and publish an API for orders?", shipped_vocab)
    assert expose.api_role in (ApiRole.EXPOSES_PARTNER_API, ApiRole.BIDIRECTIONAL_INTEGRATION) and not expose.ambiguity


def test_both_directions_named_is_bidirectional(shipped_vocab):
    r = c("I want to build an integration that calls the Export Subscriptions V1 API and publishes a webhook", shipped_vocab)
    assert r.intent is Intent.API_IMPLEMENTATION and r.api_role is ApiRole.BIDIRECTIONAL_INTEGRATION and not r.ambiguity


def test_integrate_with_is_consumption(shipped_vocab):
    r = c("How do I integrate with the Export Usage API?", shipped_vocab)
    assert r.intent is Intent.API_CONSUMPTION and r.api_role is ApiRole.CONSUMES_PLATFORM_API


def test_list_style_questions_are_discovery_without_forcing_a_direction(shipped_vocab):
    for text in ["What apis are available to me", "Which APIs can I use as a distributor?", "Give me the list of APIs"]:
        r = c(text, shipped_vocab)
        assert r.intent is Intent.API_DISCOVERY and r.api_role is ApiRole.UNCLEAR and not r.ambiguity, text


def test_error_code_is_troubleshooting(shipped_vocab):
    r = c("What does error API-4012 mean?", shipped_vocab)
    assert r.intent is Intent.TROUBLESHOOTING and r.api_role is ApiRole.NOT_APPLICABLE


def test_error_code_is_not_mistaken_for_the_word_api(shipped_vocab):
    # "API-4012" is blanked before the API-noun check, so "use" alone can't make this a consumption question.
    assert c("Can you use the code API-4012?", shipped_vocab).intent is not Intent.API_CONSUMPTION


def test_endpoint_path_counts_as_an_api_reference(shipped_vocab):
    r = c("Can I use /v1/trade-in/eligibility?", shipped_vocab)
    assert r.intent is Intent.API_CONSUMPTION


@pytest.mark.parametrize("text,intent", [
    ("What is the Buy-Sell business model?", Intent.BUSINESS_MODEL),
    ("Am I eligible for the NxM program?", Intent.PROGRAM_ELIGIBILITY),
    ("How do I get started as a new distributor?", Intent.ONBOARDING),
    ("What are the rules for the program?", Intent.PROGRAM_POLICY),
])
def test_business_and_program_questions_are_not_forced_into_api_categories(shipped_vocab, text, intent):
    r = c(text, shipped_vocab)
    assert r.intent is intent and r.api_role is ApiRole.NOT_APPLICABLE


def test_unrecognised_text_is_unknown_and_unclear(shipped_vocab):
    r = c("what is the weather today", shipped_vocab)
    assert r.intent is Intent.UNKNOWN and r.api_role is ApiRole.UNCLEAR and r.confidence < 0.5


def test_extraction_only_reports_things_that_are_actually_there(fixture_vocab, shipped_vocab):
    r = c("Which APIs can I implement for the Trade-In program in India?", fixture_vocab, role="distributor")
    assert r.program == "Trade-In" and r.region == "India" and r.partner_type == "Distributor"
    nothing = c("Which APIs can I implement?", fixture_vocab)
    assert nothing.program is None and nothing.region is None and nothing.partner_type is None


def test_partner_type_comes_from_the_session_role_when_the_text_names_none(shipped_vocab):
    assert c("What apis are available?", shipped_vocab, role="solution_provider").partner_type == "Solution Provider"
    assert c("What apis are available to a distributor?", shipped_vocab, role="reseller").partner_type == "Distributor"


def test_known_api_names_are_reported_as_entities(shipped_vocab):
    r = c("How do I call the Get Account API and GetPartnerDesignation?", shipped_vocab)
    assert "Get Account" in r.entities and "GetPartnerDesignation" in r.entities


def test_confidence_is_bounded_and_expansion_terms_are_capped(shipped_vocab):
    r = c("How do I invoke the API?", shipped_vocab, extra=[f"alias{i}" for i in range(30)])
    assert 0.0 <= r.confidence <= 1.0 and len(r.expansion_terms) <= 8


def test_classification_is_deterministic(shipped_vocab):
    text = "Which APIs can I implement for business trade-in?"
    assert c(text, shipped_vocab) == c(text, shipped_vocab)


def test_empty_vocabulary_yields_unknown(empty_vocab):
    r = c("Which APIs should I consume?", empty_vocab)
    assert r.intent is Intent.UNKNOWN and r.api_role is ApiRole.UNCLEAR
