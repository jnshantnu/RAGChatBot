"""The optional LLM rewrite retry (the "middle path" between the deterministic
rewriter and always invoking an LLM). No network: the model is a fake."""
import json
from types import SimpleNamespace

import pytest

from retrieval.query_understanding.llm_rewrite import (
    LlmRewriteError, build_messages, parse_rewrite, rewrite_with_llm,
)


class FakeClient:
    """Stands in for the OpenAI SDK client."""
    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def reply(rewritten):
    return json.dumps({"rewritten_query": rewritten})


# ---------- the model call itself ----------

def test_a_clean_rewrite_is_returned():
    client = FakeClient(reply("which apis can a distributor use"))
    out = rewrite_with_llm("which aisp can a distributor use in philipines", client=client, model="m", timeout=2.5)
    assert out == "which apis can a distributor use"
    call = client.calls[0]
    assert call["model"] == "m" and call["timeout"] == 2.5 and call["temperature"] == 0


def test_prompt_sends_only_the_query_text():
    user = build_messages("which aisp can a distributor use")[1]["content"]
    assert user == "Query: which aisp can a distributor use"


@pytest.mark.parametrize("content", [
    "", "not json", "{}", '{"rewritten_query": "ok"}'[:-1],
])
def test_off_contract_replies_are_errors(content):
    with pytest.raises(LlmRewriteError):
        rewrite_with_llm("a real question", client=FakeClient(content), model="m", timeout=1)


def test_a_too_short_rewrite_is_rejected():
    # a rewrite of one word has thrown the question away, not cleaned it up
    with pytest.raises(LlmRewriteError, match="too short"):
        parse_rewrite(reply("apis"), "which aisp can a distributor use in philipines")


def test_a_wildly_longer_rewrite_is_rejected_as_a_likely_hallucination():
    original = "which apis can I use"
    bloated = " ".join(["extra"] * 20) + " " + original
    with pytest.raises(LlmRewriteError, match="longer"):
        parse_rewrite(reply(bloated), original)


def test_an_unchanged_rewrite_is_accepted_not_an_error():
    # "I couldn't improve it" is a valid, honest answer -- not a failure
    assert parse_rewrite(reply("which apis can I use"), "which apis can I use") == "which apis can I use"


def test_transport_errors_become_rewrite_errors():
    with pytest.raises(LlmRewriteError, match="TimeoutError"):
        rewrite_with_llm("a real question", client=FakeClient(error=TimeoutError("read timed out")), model="m", timeout=1)


def test_missing_api_key_is_an_error_not_a_crash(monkeypatch):
    import config
    from retrieval.query_understanding import llm_rewrite

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm_rewrite, "_default_client", None)
    with pytest.raises(LlmRewriteError, match="OPENROUTER_API_KEY"):
        rewrite_with_llm("a real question")
