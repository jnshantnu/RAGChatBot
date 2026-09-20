from llm import generate
from retrieval.query_understanding.schema import ApiRole, Intent, QueryUnderstandingResult
from retrieval.rrf import FusedResult


def chunk():
    return FusedResult(chunk_id="c1", doc_id="doc", heading="heading", chunk_text="body text", fused_score=0.1, arms=["keyword"])


def understanding(**kw):
    base = dict(
        original_query="which APIs can I implement for biz trade in?",
        normalized_query="Which APIs can I implement for business trade-in?",
        retrieval_query="Which APIs can I implement for business trade-in?",
        intent=Intent.API_IMPLEMENTATION, api_role=ApiRole.UNCLEAR, ambiguity=True,
        clarifying_question="Do you want to consume platform APIs, or expose one?", confidence=0.5,
    )
    base.update(kw)
    return QueryUnderstandingResult(**base)


def user_prompt(**kw):
    return generate._messages("Question text", [chunk()], "reseller", **kw)[1]["content"]


def test_understanding_block_carries_original_normalized_intent_role_and_ambiguity():
    prompt = user_prompt(understanding=understanding())
    assert "Original question: which APIs can I implement for biz trade in?" in prompt
    assert "Normalized wording: Which APIs can I implement for business trade-in?" in prompt
    assert "Intent: api_implementation" in prompt and "API role: unclear" in prompt and "Ambiguous: yes" in prompt
    assert "Clarifying question: Do you want to consume platform APIs, or expose one?" in prompt


def test_clarifying_question_is_only_shown_when_ambiguous():
    prompt = user_prompt(understanding=understanding(ambiguity=False, api_role=ApiRole.CONSUMES_PLATFORM_API))
    assert "Ambiguous: no" in prompt and "Clarifying question" not in prompt


def test_normalized_line_is_omitted_when_nothing_changed():
    same = "which APIs can I consume?"
    prompt = user_prompt(understanding=understanding(original_query=same, normalized_query=same, retrieval_query=same, ambiguity=False))
    assert "Normalized wording" not in prompt


def test_a_fallback_understanding_adds_nothing_to_the_prompt():
    assert "Query understanding" not in user_prompt(understanding=QueryUnderstandingResult.fallback("q"))
    assert "Query understanding" not in user_prompt(understanding=None)
    assert "Query understanding" not in user_prompt()


def test_prompt_order_is_role_then_history_then_understanding_then_context_then_question():
    prompt = user_prompt(history=[{"query": "prior q", "answer": "prior a [3]"}], understanding=understanding())
    positions = [prompt.index(s) for s in ("Session role:", "Earlier in this conversation:", "Query understanding", "Context:", "Question: Question text")]
    assert positions == sorted(positions)
    assert "[3]" not in prompt.split("Context:")[0]                # stale citation markers are stripped from history


def test_system_prompt_states_the_answer_rules():
    p = generate.SYSTEM_PROMPT
    assert "ORIGINAL question is authoritative" in p
    assert 'never turn "consume" into "implement" or "expose"' in p
    assert "Assuming you mean APIs your application calls from our platform" in p
    assert "Assuming you mean an API or webhook your system exposes" in p
    assert "ask the clarifying question" in p and "BEFORE recommending any specific API" in p
    assert "Every claim must end with a citation" in p and "I don't have that information." in p   # existing contract intact
    assert "never introduce one that isn't there" in p                                              # grounding rule intact


def test_original_question_stays_authoritative_when_the_question_line_is_a_rewrite():
    prompt = user_prompt(understanding=understanding())
    assert prompt.index("Original question:") < prompt.index("Question: Question text")
