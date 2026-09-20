"""Context assembly + LLM call + citation verification.

The permission refusal path (asking about content the session's ACL excludes)
never reaches here: the retriever already filtered those chunks out at the SQL
level, so the LLM only ever sees chunks the user is allowed to see. If nothing
relevant survives retrieval, the confidence gate abstains before this module
is even called -- "I don't have that" is decided upstream, not by the model.

Every chunk handed to generate_answer is citable -- including chunks marked
`metadata.guaranteed=True` (e.g. ADSK-BIZ-RULES.md), which retrieval/pipeline.py
fetches unconditionally so they can never lose the top-20 rerank competition
and silently vanish from an answer. "Guaranteed" only affects how a chunk gets
into the context here, not whether the model may cite it.

ACL only controls which chunks are retrieved at all (document-level: public
vs. restricted), not role-specific content *within* a chunk that's visible to
everyone -- e.g. ADSK-BIZ-RULES.md's partner-tier section covers Distributor
and Reseller in the same paragraph, since both can see that doc. So the
model is told the session's role explicitly here; without it, "which APIs
are available to me" has no way to resolve who "me" is, and the model would
reasonably explain every role's access instead of just the asker's.
"""
import re
from dataclasses import dataclass

from openai import OpenAI

import config
from retrieval.rrf import FusedResult

_client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)

SYSTEM_PROMPT = """You are a partner-support assistant. Every fact, value, \
endpoint, parameter, or field name in your answer must come from the numbered \
context chunks provided below -- never introduce one that isn't there, even \
if it seems standard or obvious.

You may translate those facts into a different format the user asks for -- \
for example, expressing a documented request/response flow as working code, \
a table, or a checklist. That is not outside knowledge: the syntax and \
structure are yours to construct, but every concrete detail inside it (URLs, \
header names, field names, encoding, parameter values) must still trace back \
to the context. If completing what's asked would require a detail the \
context doesn't give you (e.g. a specific SDK method, a config value), say so \
explicitly instead of filling it in yourself.

Every claim must end with a citation like [1] or [2] referencing the chunk \
number it came from. For synthesized content like code, cite the chunk(s) its \
facts were drawn from in the explanation around it, not inside the code \
itself.

If the numbered chunks don't contain enough information to answer -- even \
after allowing for format translation -- reply exactly: "I don't have that \
information."

Earlier turns in this conversation may be shown below, before the context. \
Use them ONLY to resolve what the user is referring to (e.g. "idea #1" \
meaning something you listed earlier) -- never as a source of facts. Every \
fact in your answer must still come from the numbered context given for \
THIS question, even when the question is a follow-up. If resolving the \
reference still leaves you without enough grounded information to answer, \
say so exactly as instructed above -- do not answer from what you said \
earlier alone.

A "Query understanding" block may appear before the context. The user's \
ORIGINAL question is authoritative: the normalized wording, the intent and the \
API role are only retrieval aids and can be wrong -- never change what the user \
actually asked (for example never turn "consume" into "implement" or "expose"). \
When the API role is consumes_platform_api, state your interpretation first: \
"Assuming you mean APIs your application calls from our platform...". When it is \
exposes_partner_api, state it first: "Assuming you mean an API or webhook your \
system exposes...". When the block says the question is ambiguous and the \
context does not decisively settle which meaning is intended, ask the clarifying \
question given there BEFORE recommending any specific API; you may add a brief, \
clearly separated summary of each path only if the context supports both. If the \
context supports only PART of what was asked, answer that part, say plainly what \
information is missing, and ask one concise clarifying question only if the missing \
piece is something the user could supply. (If the context supports none of it, the \
exact reply above still applies.)

The user's session role is given below, before the context. When asked what \
is available to the user (e.g. "what APIs are available to me"), answer only \
for that specific role -- list ONLY the items the context confirms that role \
can actually use. Do not list an item just to note that it's restricted or \
unavailable to them -- omit it entirely, as if it were never part of the \
context. Never describe another role's access (e.g. don't add an "if you are \
a Distributor" section when the session role is Reseller) unless the user \
explicitly asks about that other role by name."""


@dataclass
class GeneratedAnswer:
    answer: str
    cited_chunk_ids: list[str]
    uncited_claims_flagged: bool  # True if model produced text with no citation markers at all


def _build_context_block(chunks: list[FusedResult]) -> str:
    # Numbers each chunk [1], [2], ... in the order given -- these numbers are
    # what the model cites with, and what cited_chunk_ids below maps back from.
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (doc: {c.doc_id} / {c.heading})\n{c.chunk_text}")
    return "\n\n".join(lines)


def _strip_citations(text: str) -> str:
    # A prior turn's [N] markers referred to THAT turn's own numbered
    # context, which no longer exists -- this turn's context is renumbered
    # from 1 again (see _build_context_block), so an old [3] left sitting
    # next to a new, unrelated [3] would look like it's citing the new
    # context. Stripped before history ever reaches the prompt.
    return re.sub(r"\s*\[\d+\]", "", text)


def _build_history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    turns = []
    for turn in history:
        turns.append(f"User: {turn['query']}\nAssistant: {_strip_citations(turn['answer'])}")
    return "Earlier in this conversation:\n" + "\n\n".join(turns) + "\n\n"


def _build_understanding_block(understanding) -> str:
    """The query-understanding stage's verdict, shown to the model as a hint.
    Left out when the stage fell back (it decided nothing) -- an all-unknown
    block would just be noise. Original wording comes first and is labelled
    authoritative; see SYSTEM_PROMPT for how the model is told to use this."""
    if understanding is None or understanding.is_fallback:
        return ""
    lines = ["Query understanding (retrieval aid -- the original question is authoritative):",
             f"- Original question: {understanding.original_query}"]
    if understanding.normalized_query != understanding.original_query:
        lines.append(f"- Normalized wording: {understanding.normalized_query}")
    lines += [f"- Intent: {understanding.intent.value}", f"- API role: {understanding.api_role.value}",
              f"- Ambiguous: {'yes' if understanding.ambiguity else 'no'}"]
    if understanding.ambiguity and understanding.clarifying_question:
        lines.append(f"- Clarifying question: {understanding.clarifying_question}")
    return "\n".join(lines) + "\n\n"


def _messages(
    query: str, citable_chunks: list[FusedResult], role: str, history: list[dict] | None = None, understanding=None
) -> list[dict]:
    # Assemble the user-turn prompt: session role, then (if this is a
    # follow-up) earlier turns for reference resolution only, then this
    # turn's own numbered citable context, then the question. The role is
    # what lets the model resolve "me" in a question like "which APIs are
    # available to me" -- see the module docstring for why ACL alone can't
    # do this.
    user_prompt = (
        f"Session role: {role}\n\n"
        f"{_build_history_block(history)}"
        f"{_build_understanding_block(understanding)}"
        f"Context:\n{_build_context_block(citable_chunks)}\n\nQuestion: {query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def finalize_answer(answer_text: str, citable_chunks: list[FusedResult]) -> GeneratedAnswer:
    # Citation verification: pull every [N] the model wrote out of the answer
    # text, keep only the ones that are actually valid chunk numbers (guards
    # against the model inventing a citation number that doesn't exist), and
    # map each back to the chunk_id it refers to. Runs on the *complete* text,
    # so the streaming path calls this once the stream has ended -- a citation
    # can't be verified from a partial answer.
    cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    valid_indices = set(range(1, len(citable_chunks) + 1))
    cited_chunk_ids = [citable_chunks[i - 1].chunk_id for i in cited_indices if i in valid_indices]

    # Diagnostic flag: true if the model produced a substantive answer with no
    # citations at all and didn't say "I don't have that" either -- worth
    # logging/reviewing, though not itself blocked by this function.
    uncited = "I don't have that information" not in answer_text and not cited_indices

    return GeneratedAnswer(
        answer=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        uncited_claims_flagged=uncited,
    )


def generate_answer(
    query: str, citable_chunks: list[FusedResult], role: str, history: list[dict] | None = None, understanding=None
) -> GeneratedAnswer:
    response = _client.chat.completions.create(
        model=config.OPENROUTER_CHAT_MODEL,
        messages=_messages(query, citable_chunks, role, history, understanding),
        temperature=0,
    )
    return finalize_answer(response.choices[0].message.content or "", citable_chunks)


def stream_answer(
    query: str, citable_chunks: list[FusedResult], role: str, history: list[dict] | None = None, understanding=None
):
    """Same request as generate_answer, but yields the answer's text as the model
    writes it instead of returning it all at once. Total generation time is the
    same; what changes is that the first words are available almost
    immediately. Callers collect the yielded text and pass it to
    finalize_answer() afterward for citation verification."""
    stream = _client.chat.completions.create(
        model=config.OPENROUTER_CHAT_MODEL,
        messages=_messages(query, citable_chunks, role, history, understanding),
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
