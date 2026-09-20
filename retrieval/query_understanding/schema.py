"""The QueryUnderstandingResult contract: everything the query-understanding
stage tells the rest of the pipeline about one user question.

Validated on construction (bad enum -> ValueError, confidence outside 0..1
-> ValueError, expansion terms capped) and frozen, because two guarantees
matter downstream: `original_query` can never be altered after the fact (the
answer model treats it as authoritative -- rewrites are retrieval aids only),
and every field is one of a known, closed set of values rather than
free-form text a later stage has to defensively re-check.

Dataclasses, not pydantic, on purpose: that's this repo's convention for
domain objects (chat.py's ChatResponse, retrieval/rrf.py's FusedResult) --
pydantic is only used at the HTTP boundary in app/web/server.py.
"""
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

MAX_EXPANSION_TERMS = 8  # spec: "ideally 3 to 8" -- more than this stops being an aid and starts diluting a search

# Set on `warnings` when the stage could not do its job and handed back the
# untouched query instead -- chat.py keys off this to decide whether the
# legacy rewriter still has to do its own typo correction.
FALLBACK_WARNING = "query_understanding_fallback"


class Intent(str, Enum):
    API_DISCOVERY = "api_discovery"
    API_CONSUMPTION = "api_consumption"
    API_IMPLEMENTATION = "api_implementation"
    API_PUBLICATION = "api_publication"
    API_AUTHENTICATION = "api_authentication"
    BUSINESS_MODEL = "business_model"
    PROGRAM_POLICY = "program_policy"
    PROGRAM_ELIGIBILITY = "program_eligibility"
    ONBOARDING = "onboarding"
    TROUBLESHOOTING = "troubleshooting"
    UNKNOWN = "unknown"


class ApiRole(str, Enum):
    CONSUMES_PLATFORM_API = "consumes_platform_api"
    EXPOSES_PARTNER_API = "exposes_partner_api"
    BIDIRECTIONAL_INTEGRATION = "bidirectional_integration"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class CorrectedTerm:
    """One audited modification: what the user typed, what it became, and why."""
    original: str
    normalized: str
    reason: str          # "abbreviation" | "synonym" | "typo"
    confidence: float    # 0..1 -- 1.0 for exact vocabulary matches, fuzzy score/100 for typos

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"CorrectedTerm.confidence must be within 0..1, got {self.confidence}")


@dataclass(frozen=True)
class QueryUnderstandingResult:
    original_query: str
    normalized_query: str
    retrieval_query: str
    intent: Intent = Intent.UNKNOWN
    api_role: ApiRole = ApiRole.UNCLEAR
    program: str | None = None
    partner_type: str | None = None
    region: str | None = None
    entities: tuple[str, ...] = ()
    expansion_terms: tuple[str, ...] = ()
    corrected_terms: tuple[CorrectedTerm, ...] = ()
    ambiguity: bool = False
    clarifying_question: str | None = None
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()
    classifier: str = "rules"  # who decided intent/api_role: "rules", "llm" (the optional fallback), or "none" (stage bypassed)

    def __post_init__(self):
        # Coerce plain strings (e.g. from JSON) to the enums; anything that
        # isn't a valid member raises rather than being carried along.
        object.__setattr__(self, "intent", Intent(self.intent))
        object.__setattr__(self, "api_role", ApiRole(self.api_role))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"QueryUnderstandingResult.confidence must be within 0..1, got {self.confidence}")
        if self.classifier not in ("rules", "llm", "none"):
            raise ValueError(f"QueryUnderstandingResult.classifier must be rules|llm|none, got {self.classifier!r}")
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "expansion_terms", tuple(self.expansion_terms)[:MAX_EXPANSION_TERMS])
        object.__setattr__(self, "corrected_terms", tuple(self.corrected_terms))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @classmethod
    def fallback(cls, original_query: str, warning: str | None = None) -> "QueryUnderstandingResult":
        """The 'do nothing, but stay functional' result: the original text is
        used for retrieval as-is and nothing is guessed. Used when the stage is
        switched off or fails -- the chatbot must never be blocked by it."""
        warnings = (FALLBACK_WARNING,) + ((warning,) if warning else ())
        return cls(
            original_query=original_query, normalized_query=original_query, retrieval_query=original_query,
            intent=Intent.UNKNOWN, api_role=ApiRole.UNCLEAR, confidence=0.0, warnings=warnings, classifier="none",
        )

    @property
    def is_fallback(self) -> bool:
        return FALLBACK_WARNING in self.warnings

    def with_updates(self, **changes) -> "QueryUnderstandingResult":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        """JSON-safe form (enums as their string values) for the API response,
        the debug trace, and logs."""
        d = asdict(self)
        d["intent"] = self.intent.value
        d["api_role"] = self.api_role.value
        return d

    def log_summary(self) -> dict:
        """Counts and categories only -- deliberately NO query text, so this can
        be logged without widening what the app's logs already contain."""
        return {
            "intent": self.intent.value,
            "api_role": self.api_role.value,
            "ambiguity": self.ambiguity,
            "confidence": round(self.confidence, 2),
            "normalization_count": len(self.corrected_terms),
            "corrected_terms_count": sum(1 for t in self.corrected_terms if t.reason == "typo"),
            "warnings_count": len(self.warnings),
            "fallback": self.is_fallback,
            "classifier": self.classifier,
        }
