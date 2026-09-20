"""Protected-term masking: spans of a query that must NEVER be rewritten,
spell-corrected or expanded -- URLs, endpoint paths, code, error codes, IDs,
version strings, HTTP methods, known API names and protected vocabulary terms.

How it's used: `mask()` swaps every protected span for an opaque sentinel
(digits between two private-use characters -- deliberately containing no
ASCII letters, so no word-level regex or fuzzy matcher can ever see a "word"
in it), the normalizer / typo corrector runs on the masked text, then
`unmask()` puts the exact original text back. That makes protection
structural rather than something each rule has to remember.

This closes a real gap: retrieval/query_rewrite.py's typo corrector and
"service" -> "APIs" synonym rule used to run over EVERY [a-zA-Z]+ run in the
query, including the inside of a URL or a code snippet.
"""
import re

from retrieval.query_understanding.vocabulary import Vocabulary

_SENTINEL_OPEN, _SENTINEL_CLOSE = "\ue000", "\ue001"  # private-use code points, written as escapes so they stay visible in editors
_SENTINEL_RE = re.compile(f"{_SENTINEL_OPEN}(\\d+){_SENTINEL_CLOSE}")

# (kind, pattern) in priority order -- earlier kinds win an overlap.
_STRUCTURAL = [
    ("code_block", re.compile(r"```.*?```", re.DOTALL)),
    ("inline_code", re.compile(r"`[^`\n]+`")),
    ("url", re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("uuid", re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")),
    ("endpoint_path", re.compile(r"(?<![\w/:.])/[A-Za-z0-9_.~%{}:-]+(?:/[A-Za-z0-9_.~%{}:-]*)*")),
    ("endpoint_path", re.compile(r"(?<![\w/])[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.~%{}:-]+){2,}")),
    ("error_code", re.compile(r"\b[A-Za-z]{1,10}[-_]\d{2,6}\b")),
    ("version", re.compile(r"(?<![\w.])v\d+(?:\.\d+){0,3}(?!\w)", re.IGNORECASE)),
    ("version", re.compile(r"(?<![\w.])\d+\.\d+(?:\.\d+)*(?!\w)")),
    ("http_method", re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")),  # uppercase only: "get" the verb is untouched
    ("identifier", re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{4,}\b")),  # SKUs, partner/program codes, keys: letters AND digits
]
_CAMEL_CASE = ("api_name", re.compile(r"(?<![\w])(?:[A-Z][a-z0-9]+){2,}(?![\w])"))  # GetSubscriptions, PlaceOrder


def find_protected_spans(text: str, vocab: Vocabulary, *, include_names: bool = True) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, kind) spans, sorted by position.

    `include_names=False` limits protection to purely structural text (URLs,
    paths, code, ids, versions, ...) -- used by the legacy "Partner
    WebServices" -> "APIs" rewrite, which must still be allowed to touch
    product-name words like WebServices/PWS that `include_names` would shield."""
    patterns = list(_STRUCTURAL)
    if include_names:
        patterns.append(_CAMEL_CASE)
    candidates = []
    for kind, pattern in patterns:
        candidates += [(m.start(), m.end(), kind) for m in pattern.finditer(text)]
    if include_names and vocab.protected_name_regex is not None:
        candidates += [(m.start(), m.end(), "protected_term") for m in vocab.protected_name_regex.finditer(text)]
    # Earliest start first; for the same start, the longest span wins.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    spans, cursor = [], 0
    for start, end, kind in candidates:
        if start >= cursor:
            spans.append((start, end, kind))
            cursor = end
    return spans


def mask(text: str, vocab: Vocabulary, *, include_names: bool = True) -> tuple[str, list[str]]:
    """Returns (masked_text, originals); pass both to `unmask` afterwards."""
    originals: list[str] = []
    out, cursor = [], 0
    for start, end, _kind in find_protected_spans(text, vocab, include_names=include_names):
        out.append(text[cursor:start])
        out.append(f"{_SENTINEL_OPEN}{len(originals)}{_SENTINEL_CLOSE}")
        originals.append(text[start:end])
        cursor = end
    out.append(text[cursor:])
    return "".join(out), originals


def unmask(text: str, originals: list[str]) -> str:
    return _SENTINEL_RE.sub(lambda m: originals[int(m.group(1))], text)


def has_kind(text: str, vocab: Vocabulary, kind: str) -> bool:
    return any(k == kind for _s, _e, k in find_protected_spans(text, vocab))


def blank_protected(text: str, vocab: Vocabulary, *, include_names: bool = True) -> str:
    """The text with every protected span replaced by a space -- what the
    intent rules read, so an error code like API-4012 can't be mistaken for
    the word "API". `include_names=False` blanks only structural spans (URLs,
    paths, codes, ids) and leaves program/API names readable -- the business-
    model rules need "Buy-Sell" and "NxM" visible."""
    masked, _ = mask(text, vocab, include_names=include_names)
    return _SENTINEL_RE.sub(" ", masked)


def api_reference_view(text: str, vocab: Vocabulary) -> str:
    """Like blank_protected, but endpoint paths, CamelCase API names and
    known API/protected terms are replaced by the word "API" instead of a
    blank -- so "use /v1/orders" or "use GetSubscriptions" read as 'use API',
    while an error code or URL still reads as nothing at all."""
    out, cursor = [], 0
    for start, end, kind in find_protected_spans(text, vocab):
        out.append(text[cursor:start])
        out.append(" API " if kind in ("endpoint_path", "api_name", "protected_term") else " ")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
