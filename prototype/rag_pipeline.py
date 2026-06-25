import os
from typing import List, Tuple, Optional

try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, VECTOR_DB_PATH
except ImportError:
    from prototype.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, VECTOR_DB_PATH

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class DocumentProcessor:
    """Handles document ingestion, chunking, embedding, and retrieval."""

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.persist_dir = os.path.join(VECTOR_DB_PATH, collection_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # Prefer splitting at paragraph/sentence boundaries before characters
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.vector_store: Optional[Chroma] = None
        self._try_load_existing()

    def _try_load_existing(self) -> None:
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
                # cosine: score range 0–1, higher = more similar
                collection_metadata={"hnsw:space": "cosine"},
            )

    def ingest_documents(self, documents: List[Document]) -> int:
        """Chunk, embed, and store documents. Returns number of chunks created."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} docs into {len(chunks)} chunks")

        if self.vector_store is not None:
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )

        print(f"Stored {len(chunks)} chunks in collection '{self.collection_name}'")
        return len(chunks)

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """
        Return top-k chunks above the similarity threshold.

        Uses cosine similarity (0–1). The fix from the original code:
        Chroma's default L2 distance returns lower=better, but with
        hnsw:space=cosine and similarity_search_with_relevance_scores,
        we get normalized 0–1 scores where higher=better, so
        score >= threshold correctly keeps the good matches.
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, score) for doc, score in results if score >= SIMILARITY_THRESHOLD]

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None
