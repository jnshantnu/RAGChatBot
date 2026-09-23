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

A second, more serious typo gap surfaced later: "authenication" (missing a
letter) isn't covered by any of the three curated word lists above, so it
reached retrieval unrewritten. Retrieval and RRF fusion handled it fine
(the right page landed 3rd in the fused top-20, embeddings don't care about
a typo) -- but the cross-encoder reranker (retrieval/rerank.py) scored every
candidate near-zero for the typo'd query (top score 0.0017) vs. a real
answer (0.3315, clears the 0.20 confidence gate) for the exact same
candidates with only that one word fixed. Measured, not assumed: fixing the
query's informal phrasing/ellipsis on its own barely moved the score
(0.0017 -> 0.0031); the typo was the entire effect. A cross-encoder judges
literal token overlap much more than an embedding does, so it's far less
forgiving of a misspelled topic word -- and "authentication" is exactly a
topic word, unlike "apis"'s incidental function-word typo, so this wasn't a
rare edge case to shrug off.

_DOMAIN_VOCAB below is the fix: instead of hand-curating a word list ahead
of time (which only ever covers typos someone thought to add), it's built
from the corpus itself (distinct chunks.doc_id/heading text, see
_build_domain_vocab) -- so it grows automatically with new documents and
catches whatever topic word gets typo'd next, not just this one.
"""
import re
import threading
from dataclasses import dataclass, field

import psycopg
from rapidfuzz import fuzz

import config
from retrieval.query_understanding import protected
from retrieval.query_understanding.normalize import correct_typos
from retrieval.query_understanding.vocabulary import close_under_plural, get_vocabulary

# The word lists below used to be hardcoded in this file; they now live in
# retrieval/query_vocabulary.json ("legacy_rewriter" section) so owners can edit
# them without code -- see docs/query-understanding.md. (If that file is
# unreadable, get_vocabulary() logs an ERROR and these lists come back empty:
# the rules simply don't fire, they never crash a request.)
_VOCAB = get_vocabulary()
_LEGACY = _VOCAB.legacy

FUZZY_THRESHOLD = _VOCAB.typo_high_confidence

# Synonym normalization: informal/technical terms partners use in place of
# "APIs" -- corpus-grounded (checked actual frequency in RAG-KB-Documents
# before including each one; generic words like "feature"/"call"/"function"
# were deliberately left out as too ambiguous to blanket-substitute).
# Multi-word phrases are matched exactly (typos spanning a whole phrase are
# rare and unreliable to fuzzy-match token-by-token); single words are
# fuzzy-matched, since that's where a typo like "servces" actually occurs.
_SYNONYM_PHRASES = list(_LEGACY.get("synonym_phrases", ()))
_SYNONYM_WORDS = list(_LEGACY.get("synonym_words", ()))

# Curated bare-term list: standalone terms that are genuinely ambiguous as a
# query on their own, expanded into a real question instead of left as-is.
# Deliberately a fixed, reviewed list -- not a general "looks like a term"
# heuristic -- so it stays predictable and only catches what's been checked.
_BARE_TERMS = list(_LEGACY.get("bare_terms", ()))

_SELF_REFERENCE = re.compile(r"\b(me|my|i)\b", re.IGNORECASE)
_AVAILABILITY_PHRASES = re.compile(r"\bcan i use\b|\bdo i have\b", re.IGNORECASE)
_AVAILABILITY_WORDS = list(_LEGACY.get("availability_words", ()))

_ROLE_LABELS = {
    "reseller": "Reseller", "distributor": "Distributor",
    "solution_provider": "Solution Provider", "principal": "Principal",
}

# Words the three curated rules above already own -- excluded from the
# corpus-derived vocabulary below so the two mechanisms never fight over the
# same word (e.g. vocabulary correction silently "fixing" a typo of
# "available" before role-injection's own fuzzy check ever sees it).
_ALREADY_HANDLED_WORDS = {
    w.lower() for phrase in _SYNONYM_PHRASES for w in phrase.split()
} | {w.lower() for w in _SYNONYM_WORDS} | {w.lower() for w in _AVAILABILITY_WORDS} | {
    w.lower() for term in _BARE_TERMS for w in re.findall(r"[a-zA-Z]+", term)
} | {"api", "apis"}

# Corpus-derived vocabulary size/precision knobs -- see the module docstring
# for why this exists (the "authenication" case) and why it's built from the
# corpus rather than hand-curated like the lists above.
_VOCAB_MIN_WORD_LEN = _VOCAB.typo_min_word_length     # same reasoning as _fuzzy_match's length guard, one notch stricter since these terms aren't individually reviewed
_VOCAB_MAX_DOC_FRACTION = 0.35  # drop boilerplate that's in more than ~1/3 of documents (e.g. "partner"/"autodesk"/"manual" show up in 16-20 of this corpus's 21 docs; real topic words like "authentication"/"subscriptions" show up in a handful)

# Hand-added verb-form companions, same curated-exception philosophy as
# _SYNONYM_WORDS/_BARE_TERMS above: the vocabulary is built from document
# TITLES, which use the noun ("API Authentication Service Reference
# Manual"), so the verb a user actually types ("authenticate") never
# appears there and can't be derived automatically -- English noun/verb
# derivation isn't a reliable suffix rule (a blind "-ation" -> "-ate" rule
# would turn "implementation" into the non-word "implementate"). Measured:
# a typo'd "authenicate" scores only 80 against "authentication" (below the
# 85 threshold, correctly -- they really are different words) and 0 against
# nothing, since "authenticate" wasn't in the vocabulary at all to score
# against. Extend this set if another noun/verb split like this is observed.
_VOCAB_MORPHOLOGICAL_EXTRAS = set(_LEGACY.get("morphological_extras", ()))  # from query_vocabulary.json

_domain_vocab: set[str] | None = None
_domain_vocab_lock = threading.Lock()  # mirrors retrieval/rerank.py's _load_lock: lazy, one-time-per-process, safe if two requests race the first build


def _build_domain_vocab() -> set[str]:
    # Vocabulary = words from chunks.doc_id/heading, which are structural
    # (document titles and page headings), not full chunk_text prose -- much
    # lower noise than mining every sentence, at the cost of not catching a
    # topic word that's discussed in a document's body but never named in
    # its title/headings. Good enough for this corpus (see the module
    # docstring's empirical check); the fix if that gap ever bites is
    # widening this query to chunk_text with a frequency BAND (not just an
    # upper bound) to separate topic nouns from incidental prose words.
    try:
        with psycopg.connect(config.DATABASE_URL) as conn:
            rows = conn.execute("SELECT DISTINCT doc_id, heading FROM chunks").fetchall()
    except Exception:
        return set()  # never let a DB hiccup break query rewriting -- see rerank.py's own preload for the same philosophy

    per_doc_words: dict[str, set[str]] = {}
    for doc_id, heading in rows:
        words = re.findall(r"[a-zA-Z]+", doc_id.replace("-", " ")) + re.findall(r"[a-zA-Z]+", heading)
        per_doc_words.setdefault(doc_id, set()).update(w.lower() for w in words)

    total_docs = len(per_doc_words) or 1
    doc_freq: dict[str, int] = {}
    for words in per_doc_words.values():
        for w in words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    max_freq = total_docs * _VOCAB_MAX_DOC_FRACTION
    vocab = {
        w for w, f in doc_freq.items()
        if len(w) >= _VOCAB_MIN_WORD_LEN and f <= max_freq and w not in _ALREADY_HANDLED_WORDS
    }

    # Close the set under trivial pluralization: "model" and "models" often
    # don't clear the same document-frequency band independently (one doc
    # says "Buy Sell Model", another says "APIs -- Model-Specific..."), so
    # only one form would survive filtering on its own -- and the OTHER,
    # perfectly-spelled form would then look like a typo of it and get
    # "corrected" into the wrong number (caught empirically: a correctly
    # typed "business model?" was rewritten to "business models?"). Adding
    # both forms once either passes means an exact match always wins before
    # fuzzy correction even runs.
    return close_under_plural(vocab, _VOCAB_MIN_WORD_LEN) | _VOCAB_MORPHOLOGICAL_EXTRAS


def _get_domain_vocab() -> set[str]:
    global _domain_vocab
    if _domain_vocab is None:
        with _domain_vocab_lock:
            if _domain_vocab is None:
                _domain_vocab = _build_domain_vocab()
    return _domain_vocab


def get_domain_vocab() -> set[str]:
    """Public accessor (chat.py hands this to the query-understanding stage so
    its typo correction can use the same corpus-derived vocabulary)."""
    return _get_domain_vocab()


def _correct_vocabulary(query: str) -> tuple[str, bool]:
    """Fuzzy-corrects a typo'd word to its exact spelling in the corpus-
    derived vocabulary -- e.g. "authenication" -> "authentication". The
    matching logic (single-hit-or-clear-lead, score threshold) lives in
    retrieval/query_understanding/normalize.py:correct_typos, shared with the
    new normalization stage so there is exactly one implementation.

    Protected spans (URLs, endpoint paths, code, error codes, IDs, version
    strings, known API names) are masked out first: before this, the
    corrector ran over EVERY letter-run in the query, including the inside of
    a URL or code snippet.
    """
    vocab = _get_domain_vocab()
    # NOTE: an empty corpus vocabulary does NOT mean there's nothing to correct
    # -- the short-word allowlist (e.g. "aips" -> "apis") needs no corpus words
    # at all, only continue past this point when vocab AND the allowlist are
    # both empty is there truly nothing correct_typos could do.
    if not vocab and not _VOCAB.typo_short_word_allowlist:
        return query, False
    masked, originals = protected.mask(query, _VOCAB)
    corrected, corrections, _warnings = correct_typos(
        masked, vocab, min_word_length=_VOCAB_MIN_WORD_LEN, high_confidence=FUZZY_THRESHOLD,
        warn_threshold=FUZZY_THRESHOLD, margin=_VOCAB.typo_margin,
        transposition=_VOCAB.typo_transposition_correction, short_allowlist=_VOCAB.typo_short_word_allowlist,
    )
    return protected.unmask(corrected, originals), bool(corrections)


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
    rewritten_query: str       # used for retrieval -- topic-focused, see _strip_code_request
    generation_query: str = ""  # used for the LLM prompt -- keeps the full request (e.g. "...in Node.js"); defaults to rewritten_query in __post_init__ when no split was needed
    rules_applied: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.generation_query:
            self.generation_query = self.rewritten_query

    @property
    def changed(self) -> bool:
        return self.original_query != self.rewritten_query


# A trailing "give me sample code in Node.js" (or similar) clause is written
# for the LLM, not for retrieval -- it names a programming language, not a
# corpus topic, and its literal wording ("sample code", the language name)
# is exactly what fooled the cross-encoder reranker into ranking a page that
# just POINTS at code samples ("visit the developer portal for JavaScript...")
# above a page that actually explains the mechanism (measured: the real page
# went from rank 8/score 0.0072 to rank 4/score 0.86 once this clause was
# left out of the reranked query -- see INTERVIEW-QA.md). Stripping it only
# for retrieval, while generation still sees the full request, is what lets
# the LLM know to write Node.js code without that same wording poisoning
# which chunks it gets to write that code from.
_CODE_REQUEST_RE = re.compile(
    r"[.,]*\s*(?:(?:give|show|send)\s+me\s+)?(?:a\s+|an\s+)?"
    r"(?:sample\s+code|code\s+sample|example\s+code|code\s+example)\s+"
    r"(?:in|for|using)\s+[a-zA-Z0-9.#+]+\.?",
    re.IGNORECASE,
)


# A "based in <region>" clause is written for the LLM, not for retrieval:
# adding a country/region name to the search text does not help it find the
# right chunk here -- it actively hurts. Measured on this corpus's own
# business-rules content: "which APIs can I use" scores 0.32 against the
# chunk that actually answers it (rerank/confidence.py's gate needs >=0.20);
# adding "in Vietnam" -- even once that exact chunk was edited to literally
# contain the word "Vietnam" -- collapsed the same pair to 0.0002. The
# cross-encoder reads an unrecognized-sounding location as a strong signal
# the question is about something else, outweighing a solid topical match on
# "which APIs can I use" (see docs/query-understanding.md). Region/partner-type
# are soft signals (item 11 of the plan): extracted and reported, but not (yet)
# used to filter or re-rank -- so this string, alone, is what was hurting
# retrieval, not a missing hard filter. Only strips a clause tied to a region
# from the controlled vocabulary (regions is a short, reviewed list -- see
# retrieval/query_vocabulary.json), never an arbitrary "in <word>", so it can't
# accidentally eat unrelated content. Generation still sees the full request
# (self.generation_query in chat.py never goes through this function), and the
# understanding stage's `region` field still reaches the prompt separately --
# nothing about knowing the user is in Vietnam is lost, only removed from the
# text used to SEARCH.
def _location_clause_pattern(regions: list[str]) -> "re.Pattern | None":
    if not regions:
        return None
    alt = "|".join(re.escape(r) for r in sorted(regions, key=len, reverse=True))
    return re.compile(
        rf"[,]*\s*if\s+(?:i(?:'m| am)|we(?:'re| are))\s+(?:based|located)\s+in\s+(?:{alt})\b\.?"
        rf"|[,]*\s*(?:i(?:'m| am)|we(?:'re| are))\s+(?:based|located)\s+in\s+(?:{alt})\b\.?",
        re.IGNORECASE,
    )


_LOCATION_CLAUSE_RE = _location_clause_pattern(list(_VOCAB.regions))


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
    # Structural spans only (URLs, paths, code, ids, versions) are shielded:
    # without this, "/v1/endpoint/status" had its "endpoint" swapped for
    # "APIs". Product-name words (WebServices, PWS) are deliberately NOT
    # shielded -- this rule exists to rewrite exactly those.
    masked, originals = protected.mask(query, _VOCAB, include_names=False)
    result, changed = _normalize_synonyms_masked(masked)
    return protected.unmask(result, originals), changed


def _normalize_synonyms_masked(query: str) -> tuple[str, bool]:
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


def rewrite_query(query: str, role: str, vocab_correction: bool = True) -> RewriteResult:
    """`vocab_correction=False` skips Rule 0 -- chat.py passes it when the
    query-understanding stage already ran typo correction on this text, so the
    same words aren't corrected twice (and the audit trail stays in one place)."""
    rules_applied = []
    working = query

    # Rule 0: corpus-vocabulary typo correction -- runs first and
    # unconditionally (like Rule 3 below), since it's general spelling
    # cleanup that every other rule's own matching should see the benefit
    # of, not a rule that competes with them for which one gets to fire.
    if vocab_correction:
        working, vocab_changed = _correct_vocabulary(working)
        if vocab_changed:
            rules_applied.append("vocabulary_correction")

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

    # Final step: split into a retrieval-focused query and a generation
    # query, only when there's an actual format-request clause to strip --
    # see _strip_code_request's own comment for why. Runs last, on the fully
    # rewritten text, so generation still benefits from the other rules
    # (typo correction, role injection) even when this one doesn't fire.
    # Both clause-stripping rules are matched independently against the SAME
    # text and their spans removed together, rather than chaining one strip
    # into the next: a query with both a code-request clause AND a location
    # clause ("give me sample code in Java if I am based in Vietnam") would
    # otherwise have the location clause left as the ENTIRE remaining text
    # after the code clause was removed, and the "never return an empty
    # string" safeguard below would then hand back the un-stripped text --
    # silently undoing the fix for the one query that needed it most.
    stripped_spans = []
    code_match = _CODE_REQUEST_RE.search(working)
    if code_match:
        stripped_spans.append(("code_request_stripped_for_retrieval", code_match))
    location_match = _LOCATION_CLAUSE_RE.search(working) if _LOCATION_CLAUSE_RE else None
    if location_match:
        stripped_spans.append(("location_clause_stripped_for_retrieval", location_match))

    retrieval_text = working
    for _, m in sorted(stripped_spans, key=lambda t: t[1].start(), reverse=True):  # rightmost first, so earlier spans keep their indices
        retrieval_text = retrieval_text[: m.start()] + retrieval_text[m.end() :]
    retrieval_text = re.sub(r"([.!?])\s*,\s*", r"\1 ", retrieval_text)  # an orphaned comma left directly after a sentence end
    retrieval_text = re.sub(r"[.\s]+$", "", retrieval_text).strip()
    if retrieval_text:
        rules_applied += [name for name, _ in stripped_spans]
    else:
        retrieval_text = working  # stripping everything would leave nothing to search on; keep the original instead

    return RewriteResult(
        original_query=query, rewritten_query=retrieval_text, generation_query=working, rules_applied=rules_applied
    )
