import pytest

from retrieval.query_understanding import protected


def kinds(text, vocab, **kw):
    return {(text[s:e], k) for s, e, k in protected.find_protected_spans(text, vocab, **kw)}


@pytest.mark.parametrize("text,protected_text,kind", [
    ("see https://api.example.com/v1/orders?id=7 now", "https://api.example.com/v1/orders?id=7", "url"),
    ("mail me at partner.ops@example.com please", "partner.ops@example.com", "email"),
    ("call /v1/trade-in/eligibility today", "/v1/trade-in/eligibility", "endpoint_path"),
    ("try v2/oauth/generateaccesstoken now", "v2/oauth/generateaccesstoken", "endpoint_path"),
    ("what does error API-4012 mean", "API-4012", "error_code"),
    ("code ERR_401 appeared", "ERR_401", "error_code"),
    ("uuid 123e4567-e89b-12d3-a456-426614174000 here", "123e4567-e89b-12d3-a456-426614174000", "uuid"),
    ("on version v2.1.0 only", "v2.1.0", "version"),
    ("use POST for that", "POST", "http_method"),
    ("sku AB-1234X is out", "AB-1234X", "identifier"),
    ("run `npm install foo` first", "`npm install foo`", "inline_code"),
    ("the GetSubscriptions call", "GetSubscriptions", "api_name"),
])
def test_structural_and_named_spans_are_detected(shipped_vocab, text, protected_text, kind):
    assert (protected_text, kind) in kinds(text, shipped_vocab)


def test_fenced_code_blocks_are_protected_whole(shipped_vocab):
    text = "example:\n```js\nconst servces = auth();\n```\nthanks"
    assert ("```js\nconst servces = auth();\n```", "code_block") in kinds(text, shipped_vocab)


def test_known_api_names_and_protected_terms_are_protected_case_insensitively(shipped_vocab):
    found = kinds("how to call get account and OAuth and buy-sell", shipped_vocab)
    assert ("get account", "protected_term") in found
    assert ("OAuth", "protected_term") in found
    assert ("buy-sell", "protected_term") in found


def test_plain_words_are_not_protected(shipped_vocab):
    assert kinds("which APIs can I implement for the business program", shipped_vocab) == set()
    assert kinds("and/or yes/no", shipped_vocab) == set()  # a bare slash inside a word is not a path


def test_lowercase_verbs_are_not_mistaken_for_http_methods(shipped_vocab):
    assert kinds("please get the post and delete it", shipped_vocab) == set()


def test_include_names_false_only_shields_structural_text(shipped_vocab):
    text = "Partner WebServices at https://x.io/a"
    assert ("WebServices", "api_name") in kinds(text, shipped_vocab)
    only_structural = kinds(text, shipped_vocab, include_names=False)
    assert ("https://x.io/a", "url") in only_structural and not any(k == "api_name" for _t, k in only_structural)


@pytest.mark.parametrize("text", [
    "plain sentence with nothing special",
    "GET /v1/orders/123 returns error API-4012 for uuid 123e4567-e89b-12d3-a456-426614174000",
    "```code``` and `inline` and https://a.b/c and x@y.io and GetSubscriptions",
    "",
])
def test_mask_then_unmask_is_lossless(shipped_vocab, text):
    masked, originals = protected.mask(text, shipped_vocab)
    assert protected.unmask(masked, originals) == text


def test_masked_text_contains_no_letters_from_protected_spans(shipped_vocab):
    masked, originals = protected.mask("error API-4012 at /v1/endpoint/status", shipped_vocab)
    assert "API" not in masked and "endpoint" not in masked and len(originals) == 2


def test_blank_protected_replaces_spans_with_spaces(shipped_vocab):
    assert "API-4012" not in protected.blank_protected("What does error API-4012 mean?", shipped_vocab)
