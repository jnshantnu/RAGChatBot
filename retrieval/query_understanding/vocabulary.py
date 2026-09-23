"""Loads and validates the controlled vocabulary (retrieval/query_vocabulary.json).

Everything that used to be a hardcoded word list scattered through
retrieval/query_rewrite.py -- plus the new abbreviation/synonym/intent data --
lives in that one JSON file, so product/program owners can extend it without
touching code. This module is the only thing that reads the file: it parses
it into an immutable `Vocabulary` (with regexes pre-compiled once) and
rejects anything malformed with a message that names the offending entry.

`get_vocabulary()` is the runtime entry point: loaded once per process,
cached. A missing or invalid file must not take the chatbot down, so it logs
an ERROR and returns an empty vocabulary (every stage then no-ops and the
raw query is used) -- tests/test_vocabulary.py separately asserts the SHIPPED
file is valid, so a bad edit is caught by `pytest`, not discovered in prod.
"""
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field

from retrieval.query_understanding.schema import ApiRole, Intent

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "query_vocabulary.json")


def plural_partners(word: str, min_length: int) -> set[str]:
    """The singular<->plural partner(s) of `word`, using real English endings
    (opportunity <-> opportunities, business <-> businesses, webhooks <-> webhook)
    -- never a fabricated non-word like "opportunitys", which would itself
    become a fuzzy-match target and attract wrong "corrections"."""
    w = word.lower()
    if len(w) < min_length:
        return set()  # too short to be a fuzzy-match candidate at all
    if w.endswith("ies"):
        partners = {w[:-3] + "y"}
    elif w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        partners = {w[:-2]}
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        partners = {w[:-1]}
    elif w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        partners = {w[:-1] + "ies"}
    elif w.endswith(("s", "x", "z", "ch", "sh")):
        partners = {w + "es"}
    else:
        partners = {w + "s"}
    return {p for p in partners if len(p) >= min_length}


def close_under_plural(words, min_length: int) -> set[str]:
    """Each word plus its singular/plural partner. A word listed in only one
    number would otherwise look like a "typo" of its partner ("webhooks" vs
    "webhook" scores 93) and be corrected to the wrong number."""
    closed = set(words)
    for w in list(closed):
        closed |= plural_partners(w, min_length)
    return closed


class VocabularyError(ValueError):
    """The vocabulary file is malformed -- the message names the bad entry."""


@dataclass(frozen=True)
class SynonymRule:
    phrase: str
    replacement: str
    not_before: tuple[str, ...] = ()
    regex: re.Pattern = field(default=None, compare=False, repr=False)
    plural_api: bool = False  # the phrase ends in "API" -> also match/keep a plural "APIs"


@dataclass(frozen=True)
class Program:
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    abbreviations: dict = field(default_factory=dict)          # lowercase abbreviation -> expansion
    synonyms: tuple = ()                                       # tuple[SynonymRule]
    action_concepts: dict = field(default_factory=dict)        # concept -> phrases (used for expansion terms)
    action_verbs: dict = field(default_factory=dict)           # role -> verbs (used by the rules classifier)
    api_nouns: tuple = ()
    protected_terms: tuple = ()
    known_apis: tuple = ()
    known_programs: tuple = ()                                 # tuple[Program]
    partner_types: dict = field(default_factory=dict)          # lowercase phrase -> label
    regions: tuple = ()
    intent_keywords: dict = field(default_factory=dict)        # Intent value -> keywords ("stem*" = prefix match)
    clarifying_questions: dict = field(default_factory=dict)
    append_role_phrase: bool = False
    role_phrases: dict = field(default_factory=dict)           # ApiRole value -> phrase
    typo_min_word_length: int = 5
    typo_high_confidence: int = 85
    typo_warn_threshold: int = 78
    typo_margin: int = 3
    typo_warn_min_word_length: int = 7
    typo_transposition_correction: bool = True                 # fix swapped-letter typos ("dahsbaords") -- see normalize.correct_typos
    typo_short_word_allowlist: tuple = ()                      # short (4+ letter) words a swapped-letter typo may be corrected TO ("apis")
    legacy: dict = field(default_factory=dict)                 # the older rewriter lists, see query_rewrite.py
    abbreviation_regex: re.Pattern = field(default=None, compare=False, repr=False)
    protected_name_regex: re.Pattern = field(default=None, compare=False, repr=False)
    _cache: dict = field(default_factory=dict, compare=False, repr=False)  # memoised derived data (the dataclass is frozen; the dict's contents aren't)

    def candidate_words(self) -> set[str]:
        if "candidate_words" not in self._cache:
            self._cache["candidate_words"] = self._compute_candidate_words()
        return self._cache["candidate_words"]

    def _compute_candidate_words(self) -> set[str]:
        """Single lowercase words that typo correction may correct TO: the
        vocabulary's own canonical spellings. Deliberately built from
        controlled vocabulary and known entities, not a generic dictionary
        (a generic spellchecker 'fixes' API names and program codes)."""
        sources = list(self.abbreviations.values())
        sources += [s.replacement for s in self.synonyms]
        for verbs in self.action_verbs.values():
            sources += verbs
        sources += list(self.known_apis) + list(self.protected_terms) + list(self.regions)
        for p in self.known_programs:
            sources.append(p.name)
            sources += p.aliases
        sources += list(self.partner_types.values())
        words = set()
        for text in sources:
            for w in re.findall(r"[A-Za-z]+", text):
                if len(w) >= self.typo_min_word_length:
                    words.add(w.lower())
        # Close under trivial pluralisation. Without this a perfectly valid
        # "webhooks" is a fuzzy near-miss of the candidate "webhook" (93.3) and
        # gets "corrected" to the wrong number -- caught by the retrieval
        # eval, where it dropped a top rerank score from 0.37 to 0.25.
        return close_under_plural(words, self.typo_min_word_length)


# ── parsing / validation ────────────────────────────────────────────────

def _fail(msg: str):
    raise VocabularyError(msg)


def _str_list(data: dict, key: str) -> tuple:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        _fail(f"'{key}' must be a list of non-empty strings")
    return tuple(value)


def _str_map(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in value.items()
    ):
        _fail(f"'{key}' must be an object of non-empty strings")
    return value


def _phrase_regex(phrase: str, *, plural_api: bool = False, not_before: tuple = ()) -> re.Pattern:
    """Whole-phrase, case-insensitive, whitespace-tolerant, never matching
    inside a hyphenated/underscored token ('dev' must not hit 'dev-ops')."""
    tokens = phrase.split()
    parts = [re.escape(t) for t in tokens]
    if plural_api:
        parts[-1] = r"APIs?"
    body = r"\s+".join(parts)
    tail = r"(?![\w-])"
    if not_before:
        tail += r"(?!\s+(?:" + "|".join(re.escape(w) for w in not_before) + r")\b)"
    return re.compile(rf"(?<![\w-]){body}{tail}", re.IGNORECASE)


def parse_vocabulary(data: dict) -> Vocabulary:
    if not isinstance(data, dict):
        _fail("vocabulary root must be a JSON object")

    abbreviations = {k.lower(): v for k, v in _str_map(data, "abbreviations").items()}

    synonyms = []
    raw_syn = data.get("synonyms", [])
    if not isinstance(raw_syn, list):
        _fail("'synonyms' must be a list")
    for i, entry in enumerate(raw_syn):
        if not isinstance(entry, dict) or not isinstance(entry.get("phrase"), str) or not isinstance(entry.get("replacement"), str):
            _fail(f"synonyms[{i}] needs string 'phrase' and 'replacement'")
        not_before = entry.get("not_before", [])
        if not isinstance(not_before, list) or not all(isinstance(w, str) for w in not_before):
            _fail(f"synonyms[{i}].not_before must be a list of strings")
        plural_api = entry["phrase"].split()[-1].lower() == "api"
        synonyms.append(SynonymRule(
            phrase=entry["phrase"], replacement=entry["replacement"], not_before=tuple(not_before),
            regex=_phrase_regex(entry["phrase"], plural_api=plural_api, not_before=tuple(not_before)),
            plural_api=plural_api,
        ))
    # Longest phrase first, so "integrate with API" is tried before any shorter overlap.
    synonyms.sort(key=lambda s: len(s.phrase), reverse=True)

    action_concepts = {}
    for name, phrases in (data.get("action_concepts") or {}).items():
        if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
            _fail(f"action_concepts.{name} must be a list of strings")
        action_concepts[name] = tuple(phrases)
    action_verbs = {}
    for name, verbs in (data.get("action_verbs") or {}).items():
        if not isinstance(verbs, list) or not all(isinstance(v, str) for v in verbs):
            _fail(f"action_verbs.{name} must be a list of strings")
        action_verbs[name] = tuple(v.lower() for v in verbs)

    protected_terms = _str_list(data, "protected_terms")
    known_apis = _str_list(data, "known_apis")

    programs = []
    for i, entry in enumerate(data.get("known_programs", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            _fail(f"known_programs[{i}] needs a string 'name'")
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            _fail(f"known_programs[{i}].aliases must be a list of strings")
        programs.append(Program(name=entry["name"], aliases=tuple(a.lower() for a in aliases)))

    intent_keywords = {}
    valid_intents = {i.value for i in Intent}
    for name, words in (data.get("intent_keywords") or {}).items():
        if name not in valid_intents:
            _fail(f"intent_keywords.{name}: not a known intent (valid: {sorted(valid_intents)})")
        if not isinstance(words, list) or not all(isinstance(w, str) and w.strip() for w in words):
            _fail(f"intent_keywords.{name} must be a list of non-empty strings")
        intent_keywords[name] = tuple(w.lower() for w in words)

    retrieval = data.get("retrieval") or {}
    if not isinstance(retrieval.get("append_role_phrase", False), bool):
        _fail("retrieval.append_role_phrase must be true or false")
    role_phrases = retrieval.get("role_phrases", {})
    valid_roles = {r.value for r in ApiRole}
    if not isinstance(role_phrases, dict) or any(k not in valid_roles or not isinstance(v, str) for k, v in role_phrases.items()):
        _fail(f"retrieval.role_phrases keys must be api_role values {sorted(valid_roles)} with string phrases")

    typo = data.get("typo_correction") or {}
    min_len = typo.get("min_word_length", 5)
    high = typo.get("high_confidence", 85)
    warn = typo.get("warn_threshold", 78)
    margin = typo.get("margin", 3)
    warn_min_len = typo.get("warn_min_word_length", 7)
    transposition = typo.get("transposition_correction", True)
    short_allowlist = typo.get("short_word_allowlist", [])
    if not (isinstance(min_len, int) and min_len >= 3):
        _fail("typo_correction.min_word_length must be an integer >= 3")
    if not (isinstance(high, (int, float)) and isinstance(warn, (int, float)) and 0 <= warn <= high <= 100):
        _fail("typo_correction needs 0 <= warn_threshold <= high_confidence <= 100")
    if not (isinstance(margin, (int, float)) and margin >= 0):
        _fail("typo_correction.margin must be >= 0")
    if not isinstance(transposition, bool):
        _fail("typo_correction.transposition_correction must be true or false")
    if not (isinstance(short_allowlist, list) and all(isinstance(w, str) and w.isalpha() and len(w) >= 4 for w in short_allowlist)):
        _fail("typo_correction.short_word_allowlist must be a list of words of 4+ letters (shorter words collide with real words)")
    if not (isinstance(warn_min_len, int) and warn_min_len >= min_len):
        _fail("typo_correction.warn_min_word_length must be an integer >= min_word_length")

    legacy = data.get("legacy_rewriter") or {}
    s2a = legacy.get("synonym_to_apis") or {}
    legacy_norm = {
        "synonym_phrases": _str_list(s2a, "phrases"),
        "synonym_words": _str_list(s2a, "words"),
        "bare_terms": _str_list(legacy, "bare_terms"),
        "availability_words": _str_list(legacy, "availability_words"),
        "morphological_extras": set(w.lower() for w in _str_list(legacy, "morphological_extras")),
    }

    abbreviation_regex = None
    if abbreviations:
        alt = "|".join(re.escape(k) for k in sorted(abbreviations, key=len, reverse=True))
        # (?![\w-]|\.\w): not before "-x" or ".x" either, so "dev-ops" / "config.json" are left alone.
        abbreviation_regex = re.compile(rf"(?<![\w-])({alt})(?![\w-]|\.\w)", re.IGNORECASE)

    names = sorted(set(protected_terms) | set(known_apis), key=len, reverse=True)
    protected_name_regex = None
    if names:
        alt = "|".join(r"\s+".join(re.escape(t) for t in n.split()) for n in names)
        protected_name_regex = re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.IGNORECASE)

    return Vocabulary(
        abbreviations=abbreviations, synonyms=tuple(synonyms), action_concepts=action_concepts,
        action_verbs=action_verbs, api_nouns=tuple(w.lower() for w in _str_list(data, "api_nouns")),
        protected_terms=protected_terms, known_apis=known_apis, known_programs=tuple(programs),
        partner_types={k.lower(): v for k, v in _str_map(data, "partner_types").items()},
        regions=_str_list(data, "regions"), intent_keywords=intent_keywords,
        clarifying_questions=_str_map(data, "clarifying_questions"),
        append_role_phrase=bool(retrieval.get("append_role_phrase", False)), role_phrases=dict(role_phrases),
        typo_min_word_length=min_len, typo_high_confidence=high, typo_warn_threshold=warn, typo_margin=margin, typo_warn_min_word_length=warn_min_len,
        typo_transposition_correction=transposition, typo_short_word_allowlist=tuple(w.lower() for w in short_allowlist),
        legacy=legacy_norm, abbreviation_regex=abbreviation_regex, protected_name_regex=protected_name_regex,
    )


def load_vocabulary(path: str | None = None) -> Vocabulary:
    """Read + validate a vocabulary file. Raises VocabularyError on any problem."""
    path = path or DEFAULT_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise VocabularyError(f"cannot read vocabulary file {path}: {exc}") from exc
    return parse_vocabulary(data)


_vocabulary: Vocabulary | None = None
_vocabulary_lock = threading.Lock()  # same lazy, once-per-process pattern as rerank.py's model loader


def get_vocabulary() -> Vocabulary:
    global _vocabulary
    if _vocabulary is None:
        with _vocabulary_lock:
            if _vocabulary is None:
                try:
                    import config

                    _vocabulary = load_vocabulary(getattr(config, "QUERY_VOCABULARY_PATH", None))
                except Exception:
                    logger.exception("query vocabulary is missing or invalid; query understanding will be a no-op until it is fixed")
                    _vocabulary = parse_vocabulary({})
    return _vocabulary


def reset_vocabulary_cache() -> None:
    """For tests: force the next get_vocabulary() to re-read the file."""
    global _vocabulary
    with _vocabulary_lock:
        _vocabulary = None
