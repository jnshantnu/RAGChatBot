"""The query-understanding stage: one question in, one QueryUnderstandingResult out.

    original -> normalize (protected terms masked) -> classify (rules) -> retrieval query

Fail-safe by construction (item 3 of the plan): if this stage is switched off
(QUERY_UNDERSTANDING_ENABLED=false), the vocabulary is unusable, or anything
at all goes wrong, the caller gets a result whose normalized/retrieval query
is the untouched original and whose intent/role are unknown/unclear. The
chatbot keeps working exactly as it did before this stage existed -- it must
never be blocked by query understanding.

A future LLM classifier (deferred, see the plan) plugs in here behind its own
flag: it would only be consulted when `classify()` is unsure, and a failure
or timeout would land in the same fallback path.
"""
import logging
import time

from retrieval.query_understanding.normalize import normalize_query
from retrieval.query_understanding.rules import classify
from retrieval.query_understanding.schema import ApiRole, QueryUnderstandingResult
from retrieval.query_understanding.vocabulary import Vocabulary, get_vocabulary

logger = logging.getLogger(__name__)

# Below this, the role phrase is not appended to the retrieval query -- a
# shaky guess about direction must not steer the search.
ROLE_PHRASE_MIN_CONFIDENCE = 0.7


def _flag_enabled() -> bool:
    try:
        import config

        return bool(getattr(config, "QUERY_UNDERSTANDING_ENABLED", True))
    except Exception:
        return True


def _vocabulary_usable(vocab: Vocabulary) -> bool:
    return bool(vocab.abbreviations or vocab.synonyms or vocab.action_verbs or vocab.known_apis or vocab.intent_keywords)


def build_retrieval_query(normalized_query: str, api_role: ApiRole, confidence: float, vocab: Vocabulary) -> str:
    """Canonical retrieval text: the normalized question, plus -- only when the
    vocabulary opts in (retrieval.append_role_phrase) and the role is confidently
    known -- a short phrase naming the API direction. Off by default: the
    cross-encoder reranker is very sensitive to query wording, so this stays
    disabled until the eval (eval/run_understanding_eval.py) shows it doesn't
    move the confidence gate."""
    if not vocab.append_role_phrase or confidence < ROLE_PHRASE_MIN_CONFIDENCE:
        return normalized_query
    phrase = vocab.role_phrases.get(api_role.value)
    if not phrase or phrase.lower() in normalized_query.lower():
        return normalized_query
    return f"{normalized_query} {phrase}"


def role_phrase_suffix(result: QueryUnderstandingResult) -> str:
    """The part of retrieval_query added on top of normalized_query ('' if none)."""
    if result.retrieval_query.startswith(result.normalized_query):
        return result.retrieval_query[len(result.normalized_query):].strip()
    return ""


def understand_query(
    query: str,
    role: str | None = None,
    *,
    vocab: Vocabulary | None = None,
    corpus_vocab=None,
    enabled: bool | None = None,
    timings: dict | None = None,
) -> QueryUnderstandingResult:
    """`role` is the signed-in session's role (used only to report partner_type).
    `corpus_vocab` is an optional set -- or a zero-argument callable returning
    one -- of extra known-good words (the corpus-derived vocabulary) that typo
    correction may correct TO; injected so this module never touches the database.
    If a `timings` dict is passed it is filled with `normalization_ms` and
    `classification_ms` (for the request log)."""
    if enabled is None:
        enabled = _flag_enabled()
    if not enabled:
        return QueryUnderstandingResult.fallback(query, "query_understanding_disabled")
    try:
        vocab = vocab or get_vocabulary()
        if not _vocabulary_usable(vocab):
            return QueryUnderstandingResult.fallback(query, "vocabulary_unavailable")

        corpus_words = corpus_vocab() if callable(corpus_vocab) else corpus_vocab
        t_norm = time.perf_counter()
        normalization = normalize_query(query, vocab, corpus_words)
        if timings is not None:
            timings["normalization_ms"] = round((time.perf_counter() - t_norm) * 1000, 2)

        # The user's own wording of anything we canonicalised is a useful
        # extra search term ("biz" alongside "business"): concept terms first,
        # then these, capped at MAX_EXPANSION_TERMS by the classifier.
        aliases = []
        for term in normalization.corrected_terms:
            if term.reason in ("abbreviation", "synonym"):
                aliases += [term.normalized, term.original]
        t_cls = time.perf_counter()
        classification = classify(normalization.normalized, vocab, role, extra_expansions=aliases)
        if timings is not None:
            timings["classification_ms"] = round((time.perf_counter() - t_cls) * 1000, 2)

        return QueryUnderstandingResult(
            original_query=query,
            normalized_query=normalization.normalized,
            retrieval_query=build_retrieval_query(
                normalization.normalized, classification.api_role, classification.confidence, vocab
            ),
            intent=classification.intent,
            api_role=classification.api_role,
            program=classification.program,
            partner_type=classification.partner_type,
            region=classification.region,
            entities=classification.entities,
            expansion_terms=classification.expansion_terms,
            corrected_terms=normalization.corrected_terms,
            ambiguity=classification.ambiguity,
            clarifying_question=classification.clarifying_question,
            confidence=classification.confidence,
            warnings=normalization.warnings,
        )
    except Exception:
        # Deliberately broad: whatever broke, the user's question still gets answered from the raw text.
        logger.exception("query understanding failed; falling back to the original query")
        return QueryUnderstandingResult.fallback(query, "query_understanding_error")
