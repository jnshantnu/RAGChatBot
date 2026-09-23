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
from rapidfuzz.distance import OSA

from retrieval.query_understanding import protected
from retrieval.query_understanding.schema import CorrectedTerm
from retrieval.query_understanding.vocabulary import Vocabulary, close_under_plural

_WORD_RE = re.compile(r"[A-Za-z]+")

# Recorded confidence for a swapped-letter correction. It is a rule, not a
# fuzzy score: the typed word has exactly the letters of ONE known word.
TRANSPOSITION_CONFIDENCE = 0.95


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


def _letters_key(word: str) -> str:
    return "".join(sorted(word))


def _anagram_index(words, min_length: int) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for w in words:
        if len(w) >= min_length:
            index.setdefault(_letters_key(w), []).append(w)
    return index


def _swapped_letters_match(lower: str, index: dict[str, list[str]], max_edits: int) -> str | None:
    """The one known word that has EXACTLY the typed word's letters, reachable by at
    most `max_edits` adjacent-letter swaps (Damerau/OSA distance, where a swap
    costs 1 -- the plain fuzzy ratio charges 2, which is why "cerate" -> "create"
    only scores 83 and "dahsbaords" -> "dashboards" only 80). No letter is
    added, dropped or replaced, so this cannot turn one real word into another
    real word unless the two are anagrams; and it only fires on a UNIQUE match."""
    matches = [c for c in index.get(_letters_key(lower), ()) if c != lower and OSA.distance(lower, c) <= max_edits]
    return matches[0] if len(matches) == 1 else None


def correct_typos(
    text: str, candidates: set[str], *, min_word_length: int = 5, high_confidence: float = 85,
    warn_threshold: float = 78, margin: float = 3, warn_min_length: int = 7,
    transposition: bool = True, short_allowlist=(),
) -> tuple[str, list[CorrectedTerm], list[str]]:
    """Fuzzy-correct words to `candidates`. `text` should already have protected
    spans masked. Returns (text, corrections, warnings).

    Three tiers, in order:
      1. swapped letters (`transposition`): a word of `min_word_length`+ letters
         that is a rearrangement of exactly one candidate, at most 1 swap for
         words under 8 letters and 2 for longer ones ("dahsbaords" ->
         "dashboards", "cerate" -> "create");
      2. `short_allowlist`: words of 4 letters (below `min_word_length`, so
         never fuzzy-matched) may be corrected TO one of these few listed words,
         and only by a single swap ("aips" -> "apis"). Not a general short-word
         corrector: "items" can never become "teams", "apps" can never become "apis";
      3. the fuzzy ratio (`high_confidence` / `warn_threshold` / `margin`) for
         everything else, e.g. one wrong or missing letter in a long word."""
    allow = {w.lower() for w in short_allowlist}
    if not candidates and not allow:
        return text, [], []
    candidate_list = list(candidates)
    corrections: list[CorrectedTerm] = []
    warnings: list[str] = []
    indexes: dict[str, dict] = {}  # built lazily: most queries have no unknown word at all

    def index(name: str):
        if name not in indexes:
            indexes[name] = _anagram_index(candidates, min_word_length) if name == "candidates" else _anagram_index(allow, 4)
        return indexes[name]

    def swapped(word: str, match: str | None) -> str | None:
        if not match:
            return None
        out = _match_case(word, match)
        corrections.append(CorrectedTerm(word, out, "typo", TRANSPOSITION_CONFIDENCE))
        return out

    def repl(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower in candidates or lower in allow:
            return word  # already a known-good spelling
        if len(word) < min_word_length:
            # Too short to fuzzy-match safely -- but a single swap of a listed word ("aips" -> "apis") is unambiguous.
            if allow and len(word) >= 4:
                fixed = swapped(word, _swapped_letters_match(lower, index("allow"), 1))
                if fixed:
                    return fixed
            return word
        if transposition and candidates:
            fixed = swapped(word, _swapped_letters_match(lower, index("candidates"), 1 if len(word) < 8 else 2))
            if fixed:
                return fixed
        if not candidates:
            return word
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
        transposition=vocab.typo_transposition_correction, short_allowlist=vocab.typo_short_word_allowlist,
    )
    masked = _cosmetic_capitalisation(masked)

    return NormalizationResult(
        normalized=protected.unmask(masked, originals),
        corrected_terms=tuple(abbreviation_terms + synonym_terms + typo_terms),
        warnings=tuple(warnings),
    )
