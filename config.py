"""Central config, loaded once at import time from environment variables /
.env. Every other module reads its settings from here rather than calling
os.environ directly, so there's exactly one place that knows about .env."""
import os
from dotenv import load_dotenv

load_dotenv()

# Required -- no default, since there's no sensible fallback for "which database".
DATABASE_URL = os.environ["DATABASE_URL"]

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# OpenRouter exposes an OpenAI-compatible API surface at this base URL -- see
# ingest/embed.py and llm/generate.py, both of which use the OpenAI SDK
# pointed here instead of at api.openai.com.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
# Must match the `vector(N)` width in db/schema.sql -- see ingest/embed.py for
# why this is smaller than Qwen3-Embedding's native 4096-dim output.
OPENROUTER_EMBEDDING_DIM = int(os.environ.get("OPENROUTER_EMBEDDING_DIM", "1536"))
# Prepended to every query (not to documents) by ingest/embed.py:embed_query --
# asymmetric embedding models like Qwen3-Embedding expect this on the query
# side only.
OPENROUTER_EMBEDDING_QUERY_INSTRUCTION = os.environ.get(
    "OPENROUTER_EMBEDDING_QUERY_INSTRUCTION",
    "Given a partner-support question, retrieve relevant policy, API, and order documentation.",
)
OPENROUTER_CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")

# Where ingest/ingest.py looks for source documents (.pdf and .md files) to ingest.
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "RAG-KB-Documents")
