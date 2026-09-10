from openai import OpenAI

import config

# OpenRouter exposes an OpenAI-compatible API, so the official OpenAI SDK works
# unmodified -- just point base_url at OpenRouter instead of api.openai.com.
_client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed document chunks via OpenRouter's OpenAI-compatible /embeddings endpoint.
    No instruction prefix -- asymmetric retrieval models (e.g. Qwen3-Embedding) expect
    documents raw and reserve the instruction template for the query side.
    """
    if not texts:
        return []
    response = _client.embeddings.create(
        model=config.OPENROUTER_EMBEDDING_MODEL,
        input=texts,
        # Qwen3-Embedding's native output is 4096-dim, but pgvector's HNSW index
        # caps at 2000 -- requesting a smaller size here (Matryoshka truncation)
        # keeps the vectors indexable. Must match the `vector(N)` column width
        # in db/schema.sql.
        dimensions=config.OPENROUTER_EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """Embed a single search query. Prefixes the model's instruction template, which
    Qwen3-Embedding models expect on the query side (skipping it measurably hurts
    retrieval quality); harmless no-op wording for symmetric models like OpenAI's.
    """
    prefixed = f"Instruct: {config.OPENROUTER_EMBEDDING_QUERY_INSTRUCTION}\nQuery:{text}"
    return embed_texts([prefixed])[0]
