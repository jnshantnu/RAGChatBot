"""Rule-based query rewriter: three independent rewrite rules, applied before
both retrieval and generation. No LLM call -- every rule here is a
deterministic pattern match, chosen over an LLM rewriter for predictability
and zero added latency/cost per query (see the mind map's simplifications
table for that tradeoff). The case study's own architecture marks this step
"conditional, skippable" -- these rules only fire when a known trigger
matches; an already-clear query passes through unchanged.
"""
import re
from dataclasses import dataclass, field

# Synonym normalization: informal/technical terms partners use in place of
# "APIs" -- corpus-grounded (checked actual frequency in RAG-KB-Documents
# before including each one; generic words like "feature"/"call"/"function"
# were deliberately left out as too ambiguous to blanket-substitute).
_SYNONYMS = ["services", "service", "PWS", "Partner WebServices", "web services", "web service", "endpoints", "endpoint"]

# Curated bare-term list: standalone terms that are genuinely ambiguous as a
# query on their own, expanded into a real question instead of left as-is.
# Deliberately a fixed, reviewed list -- not a general "looks like a term"
# heuristic -- so it stays predictable and only catches what's been checked.
_BARE_TERMS = ["Co-Term", "True-up", "Term Switch", "NxM", "NBE", "Agency Model", "Buy Sell"]

_SELF_REFERENCE = re.compile(r"\b(me|my|i)\b", re.IGNORECASE)
_AVAILABILITY = re.compile(r"\b(available|access|accessible|can i use|do i have)\b", re.IGNORECASE)

_ROLE_LABELS = {
    "reseller": "Reseller", "distributor": "Distributor",
    "solution_provider": "Solution Provider", "principal": "Principal",
}


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
    stripped = query.strip().strip("?").strip()
    canonical = _BARE_TERMS_LOOKUP.get(_normalize_for_match(stripped))
    if canonical is None:
        return None
    return f"I am a partner with Autodesk and I want details on {canonical}."


def _needs_role_injection(query: str) -> bool:
    return bool(_SELF_REFERENCE.search(query) and _AVAILABILITY.search(query))


def _normalize_synonyms(query: str) -> tuple[str, bool]:
    changed = False
    result = query
    # Longest-first so "web service" matches before the bare "service" rule
    # would otherwise shadow it. Each pattern also consumes an optional
    # trailing "API"/"APIs" -- without that, "Partner WebServices API" would
    # become "APIs API" (the synonym swapped in right next to the word it's
    # a synonym for) instead of the correct single "APIs".
    for term in sorted(_SYNONYMS, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(term)}(?:\s+APIs?)?\b", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub("APIs", result)
            changed = True
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
