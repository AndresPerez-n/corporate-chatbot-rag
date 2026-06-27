"""
Retrieval smoke test — no LLM / API key required.
Verifies ingestion, embedding, and FAISS retrieval work end to end.
"""
from pathlib import Path
from langchain.schema import Document
from rag_pipeline import DocumentProcessor

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"

dp = DocumentProcessor(collection_name="test")

docs = []
for fp in MOCK_DATA_DIR.glob("*.txt"):
    docs.append(Document(page_content=fp.read_text(encoding="utf-8"),
                         metadata={"source": fp.name}))

dp.ingest_documents(docs)

queries = [
    "work from home policy",
    "rollback a database migration",
    "who leads Data & ML",
    "what is the company lunch subsidy",  # out of scope - should return nothing
]
for q in queries:
    print(f"\nQ: {q}")
    results = dp.retrieve(q, k=2)
    if not results:
        print("  (no results above threshold - out of scope)")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc.metadata['source']}: {doc.page_content[:55].strip()}...")
