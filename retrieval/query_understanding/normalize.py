"""Deterministic query normalization -- no LLM, no network, same input always
gives the same output.

Order (all on text where protected spans are masked out, see protected.py):
  1. whitespace / punctuation artifacts
  2. abbreviation expansion         (biz -> business)        exact, from the vocabulary
  3. synonym / canonical phrases    (trade in -> trade-in)   exact, from the vocabulary
  4. high-confidence typo correction (eligiblity -> eligibility)
  5. cosmetic capitalisation ("i" -> "I", first letter)
Every substantive change (2-4) is recorded as a CorrectedTerm, so a UI or a
log can explain exactly what was modified. Meaning is never restructured:
words are swapped for their canonical form, never reordered or dropped.

Typo policy: only correct TO a word in a controlled candidate set (the
vocabulary's own words + known entities, optionally plus the corpus-derived
vocabulary the caller passes in) -- never to an arbitrary dictionary word,
which is how a generic spellchecker corrupts API names. A near-miss that
isn't confident enough is left as typed and surfaced as a warning instead.
"""
import os
import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from retrieval.query_understanding import protected
from retrieval.query_understanding.schema import CorrectedTerm
from retrieval.query_understanding.vocabulary import Vocabulary, close_under_plural

_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class NormalizationResult:
    normalized: str
    corrected_terms: tuple[CorrectedTerm, ...]
    warnings: tuple[str, ...]


def _match_case(original: str, replacement: str) -> str:
    """Carry a leading capital across ("Biz" -> "Business", "biz" -> "business")."""
    if original[:1].isupper() and replacement[:1].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def clean_whitespace_and_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:?!])", r"\1", text)              # "word ," -> "word,"
    text = re.sub(r"([?!,;])\1+", r"\1", text)               # "??" -> "?"
    text = re.sub(r"([,;])(?=[A-Za-z0-9])", r"\1 ", text)   # "a,b" -> "a, b"
    text = re.sub(r"(\.{3})(?=[A-Za-z])", r"\1 ", text)      # "apis...give" -> "apis... give"
    text = re.sub(r"([?!])(?=[A-Za-z])", r"\1 ", text)
    return text


def expand_abbreviations(text: str, vocab: Vocabulary) -> tuple[str, list[CorrectedTerm]]:
    if vocab.abbreviation_regex is None:
        return text, []
    terms: list[CorrectedTerm] = []

    def repl(m: re.Match) -> str:
        original = m.group(1)
        out = _match_case(original, vocab.abbreviations[original.lower()])
        terms.append(CorrectedTerm(original, out, "abbreviation", 1.0))
        return out

    return vocab.abbreviation_regex.sub(repl, text), terms


def apply_synonyms(text: str, vocab: Vocabulary) -> tuple[str, list[CorrectedTerm]]:
    terms: list[CorrectedTerm] = []
    for rule in vocab.synonyms:
        def repl(m: re.Match, rule=rule) -> str:
            matched = m.group(0)
            out = _match_case(matched, rule.replacement)
            if rule.plural_api and matched.lower().endswith("s") and out.endswith("API"):
                out += "s"  # keep the user's plural: "invoke APIs" -> "consume APIs"
            if out == matched:
                return matched
            terms.append(CorrectedTerm(matched, out, "synonym", 1.0))
            return out

        text = rule.regex.sub(repl, text)
    return text, terms


def _same_stem(a: str, b: str) -> bool:
    """Plural/verb-form pairs ("opportunities"/"opportunity") aren't typos."""
    common = len(os.path.commonprefix([a, b]))
    return common >= min(len(a), len(b)) - 2


def correct_typos(
    text: str, candidates: set[str], *, min_word_length: int = 5, high_confidence: float = 85,
    warn_threshold: float = 78, margin: float = 3, warn_min_length: int = 7,
) -> tuple[str, list[CorrectedTerm], list[str]]:
    """Fuzzy-correct words to `candidates`. `text` should already have protected
    spans masked. Returns (text, corrections, warnings)."""
    if not candidates:
        return text, [], []
    candidate_list = list(candidates)
    corrections: list[CorrectedTerm] = []
    warnings: list[str] = []

    def repl(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if len(word) < min_word_length or lower in candidates:
            return word  # too short to fuzzy-match safely, or already a known-good spelling
        hits = process.extract(lower, candidate_list, scorer=fuzz.ratio, score_cutoff=warn_threshold, limit=3)
        if not hits:
            return word
        strong = [h for h in hits if h[1] >= high_confidence]
        # A typo'd plural scores >= threshold against BOTH its singular and
        # plural candidate ("subscritpions" -> subscriptions 92.3 / subscription
        # 88.0), so a bare "more than one hit" refusal would give up on a case
        # where the top hit is clearly right; require a clear lead instead.
        if strong and (len(strong) == 1 or strong[0][1] - strong[1][1] >= margin):
            best, score = strong[0][0], strong[0][1]
            out = _match_case(word, best)
            corrections.append(CorrectedTerm(word, out, "typo", round(score / 100, 3)))
            return out
        top, score = hits[0][0], hits[0][1]
        if len(word) < warn_min_length:
            return word  # 5-6 letter words collide too easily ("items" vs "teams") for a near-miss to mean anything
        if strong:
            warnings.append(f"ambiguous spelling '{word}' (candidates: {', '.join(h[0] for h in strong)}) left unchanged")
        elif not _same_stem(lower, top):
            warnings.append(f"possible typo '{word}' (closest known term '{top}', score {score:.0f}) left unchanged: not confident enough to correct")
        return word

    return _WORD_RE.sub(repl, text), corrections, warnings


def _cosmetic_capitalisation(text: str) -> str:
    text = re.sub(r"(?<![\w.'-])i(?![\w.-])", "I", text)  # standalone "i" / "i'm" -> "I" ("i.e." untouched)
    if text[:1].islower():
        text = text[:1].upper() + text[1:]
    return text


def normalize_query(text: str, vocab: Vocabulary, corpus_vocab: set[str] | None = None) -> NormalizationResult:
    masked, originals = protected.mask(text, vocab)

    masked = clean_whitespace_and_punctuation(masked)
    masked, abbreviation_terms = expand_abbreviations(masked, vocab)
    masked, synonym_terms = apply_synonyms(masked, vocab)
    masked, typo_terms, warnings = correct_typos(
        masked, vocab.candidate_words() | close_under_plural(corpus_vocab or (), vocab.typo_min_word_length),
        min_word_length=vocab.typo_min_word_length, high_confidence=vocab.typo_high_confidence,
        warn_threshold=vocab.typo_warn_threshold, margin=vocab.typo_margin, warn_min_length=vocab.typo_warn_min_word_length,
    )
    masked = _cosmetic_capitalisation(masked)

    return NormalizationResult(
        normalized=protected.unmask(masked, originals),
        corrected_terms=tuple(abbreviation_terms + synonym_terms + typo_terms),
        warnings=tuple(warnings),
    )
