-- RAG Chatbot POC schema
-- One chunks table, two index types (GIN keyword arm + HNSW semantic arm),
-- one document registry, mirroring the case-study architecture at POC scale.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',   -- active | stale | deleted
    category      TEXT,
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,          -- stable hash: doc_id + heading + ordinal
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal       INT NOT NULL,
    heading       TEXT,
    chunk_text    TEXT NOT NULL,                 -- raw text, never embedded directly
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    embedding     vector(1536),                  -- dim must match OPENROUTER_EMBEDDING_DIM; pgvector HNSW caps at 2000
    metadata      JSONB NOT NULL DEFAULT '{}',
    acl           TEXT[] NOT NULL DEFAULT ARRAY['public'],
    index_version INT NOT NULL DEFAULT 1
);

-- Keyword arm
CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx ON chunks USING GIN (tsv);

-- Semantic arm (cosine distance, matches OpenAI/OpenRouter embedding convention)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- ACL + versioning filters applied as predicates inside the index scan
CREATE INDEX IF NOT EXISTS chunks_acl_gin_idx ON chunks USING GIN (acl);
CREATE INDEX IF NOT EXISTS chunks_index_version_idx ON chunks (index_version);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
