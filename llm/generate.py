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

SYSTEM_PROMPT = """You are a partner-support assistant. Answer using only the \
numbered context chunks provided below. Do not use outside knowledge.

Every claim must end with a citation like [1] or [2] referencing the chunk \
number it came from.

If the numbered chunks don't contain enough information to answer, reply \
exactly: "I don't have that information."

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


def generate_answer(query: str, citable_chunks: list[FusedResult], role: str) -> GeneratedAnswer:
    # Assemble the user-turn prompt: session role, then numbered citable
    # context, then the question. The role is what lets the model resolve
    # "me" in a question like "which APIs are available to me" -- see the
    # module docstring for why ACL alone can't do this.
    user_prompt = f"Session role: {role}\n\nContext:\n{_build_context_block(citable_chunks)}\n\nQuestion: {query}"

    response = _client.chat.completions.create(
        model=config.OPENROUTER_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer_text = response.choices[0].message.content or ""

    # Citation verification: pull every [N] the model wrote out of the answer
    # text, keep only the ones that are actually valid chunk numbers (guards
    # against the model inventing a citation number that doesn't exist), and
    # map each back to the chunk_id it refers to.
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
