import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
OPENROUTER_EMBEDDING_DIM = int(os.environ.get("OPENROUTER_EMBEDDING_DIM", "1536"))
OPENROUTER_EMBEDDING_QUERY_INSTRUCTION = os.environ.get(
    "OPENROUTER_EMBEDDING_QUERY_INSTRUCTION",
    "Given a partner-support question, retrieve relevant policy, API, and order documentation.",
)
OPENROUTER_CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "RAG-KB-Documents")
