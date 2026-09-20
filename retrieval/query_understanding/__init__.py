"""Query understanding: normalization, protected terms, and intent/API-role
classification, run BEFORE retrieval. See docs/query-understanding.md."""
from retrieval.query_understanding.schema import (  # noqa: F401
    FALLBACK_WARNING, ApiRole, CorrectedTerm, Intent, QueryUnderstandingResult,
)
from retrieval.query_understanding.service import role_phrase_suffix, understand_query  # noqa: F401
