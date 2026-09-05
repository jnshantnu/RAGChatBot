"""Context assembly + LLM call + citation verification.

The permission refusal path (asking about content the session's ACL excludes)
never reaches here: the retriever already filtered those chunks out at the SQL
level, so the LLM only ever sees chunks the user is allowed to see. If nothing
relevant survives retrieval, the confidence gate abstains before this module
is even called -- "I don't have that" is decided upstream, not by the model.
"""
import re
from dataclasses import dataclass

from openai import OpenAI

import config
from retrieval.rrf import FusedResult

_client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)

SYSTEM_PROMPT = """You are a partner-support assistant. Answer ONLY using the numbered \
context chunks provided below. Every factual claim must end with a citation \
like [1] or [2] referencing the chunk number it came from. If the context does \
not contain enough information to answer, reply exactly: "I don't have that \
information." Do not use outside knowledge."""


@dataclass
class GeneratedAnswer:
    answer: str
    cited_chunk_ids: list[str]
    uncited_claims_flagged: bool  # True if model produced text with no citation markers at all


def _build_context_block(chunks: list[FusedResult]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (doc: {c.doc_id} / {c.heading})\n{c.chunk_text}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[FusedResult]) -> GeneratedAnswer:
    context_block = _build_context_block(chunks)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    response = _client.chat.completions.create(
        model=config.OPENROUTER_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer_text = response.choices[0].message.content or ""

    cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    valid_indices = set(range(1, len(chunks) + 1))
    cited_chunk_ids = [chunks[i - 1].chunk_id for i in cited_indices if i in valid_indices]

    uncited = "I don't have that information" not in answer_text and not cited_indices

    return GeneratedAnswer(
        answer=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        uncited_claims_flagged=uncited,
    )
