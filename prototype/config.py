import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings — local model, zero API cost
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
# 512 chars ≈ 128 tokens: small enough for precise retrieval,
# large enough to keep a coherent fact in one chunk.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K_RETRIEVAL = 5
# Cosine similarity threshold (0–1). Tuned empirically against the mock corpus:
# genuine matches score ~0.36+, out-of-scope queries fall below ~0.31, so 0.35
# separates them. A production system would tune this on a labeled eval set.
SIMILARITY_THRESHOLD = 0.35

# Vector DB — one sub-directory per collection (department namespace demo)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./faiss_index")

# Faithfulness self-check (RAGAS-style groundedness pass).
# A second, cheap LLM call verifies every claim in the answer is supported by the
# retrieved context. Off by default (adds ~1 LLM call of latency + cost); flip on
# globally here, or per-query via query(..., check_faithfulness=True).
FAITHFULNESS_CHECK_ENABLED = os.getenv("FAITHFULNESS_CHECK", "false").lower() == "true"
FAITHFULNESS_THRESHOLD = 0.5  # below this, the answer is flagged for the user to verify
# Cheaper model for the judge pass; falls back to LLM_MODEL if unset.
FAITHFULNESS_MODEL = os.getenv("FAITHFULNESS_MODEL", "gpt-4o-mini")

# API
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))

# Supported ingestion formats
SUPPORTED_FORMATS = [".pdf", ".txt", ".md"]
