"""Shared pytest fixtures.

Unit tests here are database-free and network-free: they run against a
vocabulary object built in-memory. `fixture_vocab` starts from the SHIPPED
vocabulary file (so verbs, intent keywords and abbreviations are the real
ones) and swaps in fictional entities -- a "Trade-In" program and a
"Trade-In Eligibility" API -- because the spec's examples (trade-in,
Eligibility API) don't exist in this corpus, and "correct a typo only if the
entity is known" has to be testable against a known entity.
"""
import json
import os

# config.py requires DATABASE_URL at import; unit tests never connect to it.
os.environ.setdefault("DATABASE_URL", "postgresql://unused:unused@localhost:5432/unused")

import pytest

from retrieval.query_understanding.vocabulary import DEFAULT_PATH, load_vocabulary, parse_vocabulary


@pytest.fixture(scope="session")
def shipped_vocab():
    """The real retrieval/query_vocabulary.json, exactly as deployed."""
    return load_vocabulary()


@pytest.fixture(scope="session")
def fixture_vocab():
    with open(DEFAULT_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    data["known_apis"] = ["Trade-In Eligibility", "Order Status"]
    data["known_programs"] = [{"name": "Trade-In", "aliases": ["trade in", "trade-in", "tradein"]}]
    data["protected_terms"] = ["OAuth"]
    return parse_vocabulary(data)


@pytest.fixture
def empty_vocab():
    return parse_vocabulary({})
