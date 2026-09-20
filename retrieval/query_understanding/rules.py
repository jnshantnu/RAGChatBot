"""Rules-based intent / API-role classifier -- deterministic, ~1ms, no LLM.

The one distinction this exists to get right: whether the partner wants to
CALL the platform's APIs (consume), or provide/publish something for the
platform to call (expose) -- and to notice, rather than guess, when a word
like "implement" doesn't say which. "Implement" is NEVER treated as
equivalent to "consume".

Precedence (first match wins), chosen so the more specific signal beats the
more generic one:
  troubleshooting > authentication > publication > consumption >
  implementation > discovery > eligibility > onboarding > business model >
  program policy > unknown
The verbs, nouns, keywords and clarifying question all come from the
vocabulary file, so owners can tune them without code.

Everything reported as "extracted" (program, region, partner type, API
names) must literally appear in the text or come from the session -- nothing
is inferred or invented. Unsure -> unknown / unclear / None.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

from retrieval.query_understanding import protected
from retrieval.query_understanding.schema import ApiRole, Intent, MAX_EXPANSION_TERMS
from retrieval.query_understanding.vocabulary import Vocabulary

_CAMEL = re.compile(r"(?<![\w])(?:[A-Z][a-z0-9]+){2,}(?![\w])")


@dataclass(frozen=True)
class Classification:
    intent: Intent
    api_role: ApiRole
    ambiguity: bool
    clarifying_question: str | None
    confidence: float
    program: str | None
    partner_type: str | None
    region: str | None
    entities: tuple[str, ...]
    expansion_terms: tuple[str, ...]


@lru_cache(maxsize=512)
def _keyword_regex(keyword: str) -> re.Pattern:
    # "eligib*" is a prefix match; anything else is a whole word/phrase.
    if keyword.endswith("*"):
        return re.compile(rf"\b{re.escape(keyword[:-1])}\w*", re.IGNORECASE)
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


def _any_keyword(keywords, text: str) -> bool:
    return any(_keyword_regex(k).search(text) for k in keywords)


def _phrase_pattern(phrase: str, separator: str) -> str:
    """Whole-phrase regex source; tokens may be separated by `separator`."""
    body = separator.join(re.escape(t) for t in re.split(r"[\s-]+" if separator != r"\s+" else r"\s+", phrase.strip()))
    return rf"(?<![\w-]){body}(?![\w-])"


def _find_entities(text: str, vocab: Vocabulary) -> list[str]:
    found = []
    lower = text.lower()
    for api in vocab.known_apis:
        if re.search(_phrase_pattern(api.lower(), r"\s+"), lower):
            found.append(api)
    for m in _CAMEL.finditer(text):
        if m.group(0) not in found:
            found.append(m.group(0))
    return found


def _find_program(text: str, vocab: Vocabulary) -> str | None:
    lower = text.lower()
    for program in vocab.known_programs:
        for name in (program.name.lower(), *program.aliases):
            if re.search(_phrase_pattern(name, r"[\s-]+"), lower):  # "buy sell" == "buy-sell"
                return program.name
    return None


def _find_partner_type(text: str, vocab: Vocabulary, session_role: str | None) -> str | None:
    lower = text.lower()
    for phrase, label in vocab.partner_types.items():
        if re.search(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", lower):
            return label
    if session_role:  # the signed-in session's role -- authoritative, never guessed from text
        return vocab.partner_types.get(session_role.replace("_", " ").lower())
    return None


def _find_region(text: str, vocab: Vocabulary) -> str | None:
    for region in vocab.regions:
        if re.search(rf"(?<!\w){re.escape(region)}(?!\w)", text, re.IGNORECASE):
            return region
    return None


def _dedupe(items, limit=MAX_EXPANSION_TERMS) -> tuple[str, ...]:
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return tuple(out[:limit])


def classify(
    normalized_query: str, vocab: Vocabulary, session_role: str | None = None, extra_expansions=(),
) -> Classification:
    blanked = protected.blank_protected(normalized_query, vocab)  # protected spans (codes, paths, API names) can't be mistaken for words
    readable = protected.blank_protected(normalized_query, vocab, include_names=False)  # names like Buy-Sell stay visible for the business rules
    api_view = protected.api_reference_view(normalized_query, vocab)
    verbs = vocab.action_verbs

    def has(role: str) -> bool:
        return any(re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", blanked, re.IGNORECASE) for v in verbs.get(role, ()))

    entities = _find_entities(normalized_query, vocab)
    program = _find_program(normalized_query, vocab)
    # An endpoint path (/v1/...) is itself a reference to an API, even though
    # it's blanked out of `blanked` so its words can't be read as verbs.
    has_api_noun = bool(entities) or protected.has_kind(normalized_query, vocab, "endpoint_path") or any(
        re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", blanked, re.IGNORECASE) for n in vocab.api_nouns
    )

    consume = has("consume")
    # "use" only says "call an API" when it's actually applied to one -- within
    # a couple of words of an API noun/reference ("use the API key", "use
    # /v1/orders"). "code which I can use to authenticate with our APIs" uses
    # "use" for the CODE, so it must not be read as a consume signal.
    api_alt = "|".join(re.escape(n) for n in vocab.api_nouns)
    weak_consume = any(
        re.search(
            rf"(?<![\w-]){re.escape(v)}(?![\w-])(?:\s+(?:the|our|a|an|this|that|these|platform|partner))*\s+(?:[\w.-]+\s+){{0,1}}(?:{api_alt})(?![\w-])",
            api_view, re.IGNORECASE,
        )
        for v in verbs.get("weak_consume", ())
    ) if api_alt else False
    expose, implement = has("expose"), has("implement")
    # Authentication cues are read from the un-blanked text: "OAuth" is a
    # protected term (so blanked away) but is exactly the cue we want.
    auth = any(
        re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", normalized_query, re.IGNORECASE) for v in verbs.get("authenticate", ())
    )
    troubleshooting = _any_keyword(vocab.intent_keywords.get("troubleshooting", ()), blanked) or protected.has_kind(
        normalized_query, vocab, "error_code"
    )
    discovery_shape = _any_keyword(vocab.intent_keywords.get("api_discovery", ()), blanked)

    intent, role, ambiguity, confidence = Intent.UNKNOWN, ApiRole.UNCLEAR, False, 0.3
    concept_terms: list[str] = []

    def direction() -> ApiRole:
        if (consume or weak_consume) and expose:
            return ApiRole.BIDIRECTIONAL_INTEGRATION
        if consume or weak_consume:
            return ApiRole.CONSUMES_PLATFORM_API
        if expose:
            return ApiRole.EXPOSES_PARTNER_API
        return ApiRole.UNCLEAR

    if troubleshooting:
        intent, role, confidence = Intent.TROUBLESHOOTING, ApiRole.NOT_APPLICABLE, 0.8
    elif auth:
        intent, role, confidence = Intent.API_AUTHENTICATION, direction(), 0.85
        concept_terms = list(vocab.action_concepts.get("authenticate", ())[:4])
    elif expose and has_api_noun and not consume:
        intent, role, confidence = Intent.API_PUBLICATION, ApiRole.EXPOSES_PARTNER_API, 0.9
        concept_terms = list(vocab.action_concepts.get("expose", ())[:3]) + ["webhook"]
    elif consume and has_api_noun and not (expose and implement):
        intent, role, confidence = Intent.API_CONSUMPTION, ApiRole.CONSUMES_PLATFORM_API, 0.9
        concept_terms = list(vocab.action_concepts.get("consume", ())[:2])
    elif (implement or (expose and consume)) and has_api_noun:
        intent = Intent.API_IMPLEMENTATION
        role = direction()
        if role is ApiRole.UNCLEAR:
            # The classic trap: "implement" doesn't say which way the API points.
            ambiguity, confidence = True, 0.5
            concept_terms = list(vocab.action_concepts.get("implement", ())[:2])
        else:
            confidence = 0.75
            key = "consume" if role is ApiRole.CONSUMES_PLATFORM_API else "expose"
            concept_terms = list(vocab.action_concepts.get(key, ())[:2])
    elif weak_consume and has_api_noun and not discovery_shape:
        intent, role, confidence = Intent.API_CONSUMPTION, ApiRole.CONSUMES_PLATFORM_API, 0.75
        concept_terms = list(vocab.action_concepts.get("consume", ())[:2])
    elif has_api_noun and discovery_shape:
        intent, role, confidence = Intent.API_DISCOVERY, ApiRole.UNCLEAR, 0.8  # a list-of-APIs question doesn't need a direction
    else:
        for name in (Intent.PROGRAM_ELIGIBILITY, Intent.ONBOARDING, Intent.BUSINESS_MODEL, Intent.PROGRAM_POLICY):
            if _any_keyword(vocab.intent_keywords.get(name.value, ()), readable):
                intent, role, confidence = name, ApiRole.NOT_APPLICABLE, 0.7
                break

    clarifying = vocab.clarifying_questions.get("api_direction") if ambiguity else None
    return Classification(
        intent=intent, api_role=role, ambiguity=ambiguity, clarifying_question=clarifying, confidence=confidence,
        program=program, partner_type=_find_partner_type(normalized_query, vocab, session_role),
        region=_find_region(normalized_query, vocab),
        entities=_dedupe(entities + ([program] if program else []), limit=20),
        expansion_terms=_dedupe([*concept_terms, *extra_expansions]),
    )
