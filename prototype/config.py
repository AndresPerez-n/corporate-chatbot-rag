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
# Cosine similarity threshold (0–1). 0.4 is intentionally permissive for a demo;
# production would tune this per-corpus with precision/recall eval.
SIMILARITY_THRESHOLD = 0.4

# Vector DB — one sub-directory per collection (department namespace demo)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chromadb_data")

# API
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))

# Supported ingestion formats
SUPPORTED_FORMATS = [".pdf", ".txt", ".md"]
