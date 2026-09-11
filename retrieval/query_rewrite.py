"""Rule-based query rewriter: three independent rewrite rules, applied before
both retrieval and generation. No LLM call -- every rule here is a
deterministic pattern match, chosen over an LLM rewriter for predictability
and zero added latency/cost per query (see the mind map's simplifications
table for that tradeoff). The case study's own architecture marks this step
"conditional, skippable" -- these rules only fire when a known trigger
matches; an already-clear query passes through unchanged.

Matching is fuzzy-tolerant (rapidfuzz), not exact-string -- a real regression:
"apis availabe for me" (typo, missing a letter) didn't trigger role injection
at all under exact matching, silently reproducing the exact "which APIs are
available to me" bug this rewriter was built to fix. FUZZY_THRESHOLD=85 was
tuned empirically: real typos of trigger/synonym/term words scored 85.7-94.7,
unrelated same-length words (e.g. "sources" vs "services") topped out at 66.7
-- a wide, safe margin, not a guessed number.

Known limitation, deliberately not patched: this only works because the
target words are long enough to carry that margin. "apis" is 4 letters --
its typo "apsi" scores 75 against "apis", but so do real, unrelated words
like "apps"/"aims"/"amps" (checked empirically). There's no threshold that
catches the typo without also mangling those words, so "apis" itself is not
a fuzzy-match target here. Left uncorrected on purpose: downstream semantic
search and the LLM have both been observed to produce correct, cited answers
despite this exact typo, so there's no failure to fix, only a rewriter trace
that doesn't show a correction it didn't need to make.
"""
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 85

# Synonym normalization: informal/technical terms partners use in place of
# "APIs" -- corpus-grounded (checked actual frequency in RAG-KB-Documents
# before including each one; generic words like "feature"/"call"/"function"
# were deliberately left out as too ambiguous to blanket-substitute).
# Multi-word phrases are matched exactly (typos spanning a whole phrase are
# rare and unreliable to fuzzy-match token-by-token); single words are
# fuzzy-matched, since that's where a typo like "servces" actually occurs.
_SYNONYM_PHRASES = ["Partner WebServices", "web services", "web service"]
_SYNONYM_WORDS = ["services", "service", "PWS", "endpoints", "endpoint"]

# Curated bare-term list: standalone terms that are genuinely ambiguous as a
# query on their own, expanded into a real question instead of left as-is.
# Deliberately a fixed, reviewed list -- not a general "looks like a term"
# heuristic -- so it stays predictable and only catches what's been checked.
_BARE_TERMS = ["Co-Term", "True-up", "Term Switch", "NxM", "NBE", "Agency Model", "Buy Sell"]

_SELF_REFERENCE = re.compile(r"\b(me|my|i)\b", re.IGNORECASE)
_AVAILABILITY_PHRASES = re.compile(r"\bcan i use\b|\bdo i have\b", re.IGNORECASE)
_AVAILABILITY_WORDS = ["available", "access", "accessible"]

_ROLE_LABELS = {
    "reseller": "Reseller", "distributor": "Distributor",
    "solution_provider": "Solution Provider", "principal": "Principal",
}


def _fuzzy_match(word: str, targets: list[str], threshold: int = FUZZY_THRESHOLD) -> bool:
    # Guard against short words: fuzzy-matching e.g. "is" against a 2-letter
    # target is meaningless noise, not a typo -- only worth checking once a
    # word is long enough that edit-distance-1/2 actually implies "same word".
    if len(word) < 4:
        return False
    return any(fuzz.ratio(word.lower(), t.lower()) >= threshold for t in targets)


def _normalize_for_match(s: str) -> str:
    return re.sub(r"[\s-]+", "", s).lower()


_BARE_TERMS_LOOKUP = {_normalize_for_match(t): t for t in _BARE_TERMS}


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    rules_applied: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original_query != self.rewritten_query


def _expand_bare_term(query: str) -> str | None:
    # Whole-query match only, not substring -- "Buy Sell APIs" is already a
    # clear question and shouldn't be flattened into a generic term lookup.
    # Exact normalized match first; fuzzy fallback catches a typo'd bare term
    # (e.g. "Co-Trem") the same way the other two rules below do -- compared
    # against the term's original spelling (just lowercased), not the
    # hyphen/space-stripped form: stripping structural characters from a
    # short term before fuzzy-comparing throws away signal the ratio needs
    # (e.g. "co-trem" vs "co-term" scores 85.7, but "cotrem" vs "coterm"
    # drops to 83.3 -- under threshold purely from removing the hyphen).
    stripped = query.strip().strip("?").strip()
    canonical = _BARE_TERMS_LOOKUP.get(_normalize_for_match(stripped))
    if canonical is None:
        for term in _BARE_TERMS:
            if fuzz.ratio(stripped.lower(), term.lower()) >= FUZZY_THRESHOLD:
                canonical = term
                break
    if canonical is None:
        return None
    return f"I am a partner with Autodesk and I want details on {canonical}."


def _needs_role_injection(query: str) -> bool:
    if not _SELF_REFERENCE.search(query):
        return False
    if _AVAILABILITY_PHRASES.search(query):
        return True
    words = re.findall(r"[a-zA-Z]+", query)
    return any(_fuzzy_match(w, _AVAILABILITY_WORDS) for w in words)


def _normalize_synonyms(query: str) -> tuple[str, bool]:
    changed = False
    result = query
    # Multi-word phrases first, exact match. Each pattern also consumes an
    # optional trailing "API"/"APIs" -- without that, "Partner WebServices
    # API" would become "APIs API" (the synonym swapped in right next to the
    # word it's a synonym for) instead of the correct single "APIs".
    for term in sorted(_SYNONYM_PHRASES, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(term)}(?:\s+APIs?)?\b", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub("APIs", result)
            changed = True

    # Single words, fuzzy-tolerant -- this is the exact bug class that broke
    # "available": a typo like "servces" or "endpont" must still normalize.
    def _replace_word(match):
        nonlocal changed
        word = match.group(0)
        if _fuzzy_match(word, _SYNONYM_WORDS):
            changed = True
            return "APIs"
        return word

    result = re.sub(r"[a-zA-Z]+", _replace_word, result)
    # Collapse a duplicate left over when the matched word sat right next to
    # a literal "API"/"APIs" (either order) -- same problem the phrase
    # patterns above solve with their optional trailing-API group, but a
    # single-word regex callback can't see its neighbor to consume it.
    result = re.sub(r"\bAPIs?\s+APIs?\b", "APIs", result, flags=re.IGNORECASE)
    return result, changed


def rewrite_query(query: str, role: str) -> RewriteResult:
    rules_applied = []
    working = query

    # Rule 1: bare-term expansion. Checked first and exclusively (elif below)
    # -- a bare term like "Co-Term" wouldn't match rule 2's trigger anyway,
    # and expanding it changes the query's shape entirely, so nothing else
    # should also apply to it.
    expanded = _expand_bare_term(working)
    if expanded:
        working = expanded
        rules_applied.append("bare_term_expansion")

    # Rule 2: role injection -- only for self-referential availability
    # questions ("which APIs are available to me"), where the retrieval
    # scoring has no way to resolve "me" without the session's actual role
    # in the query text itself (see retrieval/pipeline.py's rerank scoring).
    elif _needs_role_injection(working):
        role_label = _ROLE_LABELS.get(role, role)
        working = f"I am a {role_label} partner with Autodesk. {working}"
        rules_applied.append("role_injection")

    # Rule 3: synonym normalization -- runs independently of the above on
    # whatever the query looks like at this point, since it's a cheap
    # substitution that applies regardless of query shape.
    working, synonym_changed = _normalize_synonyms(working)
    if synonym_changed:
        rules_applied.append("synonym_normalization")

    return RewriteResult(original_query=query, rewritten_query=working, rules_applied=rules_applied)
