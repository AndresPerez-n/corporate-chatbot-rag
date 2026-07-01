import os
from typing import List, Tuple, Optional

try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, VECTOR_DB_PATH
except ImportError:
    from prototype.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, VECTOR_DB_PATH

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document


class DocumentProcessor:
    """Handles document ingestion, chunking, embedding, and retrieval."""

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.persist_dir = os.path.join(VECTOR_DB_PATH, collection_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            # Normalize vectors so inner product == cosine similarity (0-1)
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: Optional[FAISS] = None
        self._try_load_existing()

    def _try_load_existing(self) -> None:
        index_path = os.path.join(self.persist_dir, "index.faiss")
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def ingest_documents(self, documents: List[Document]) -> int:
        """Chunk, embed, and store documents. Returns number of chunks created."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} docs into {len(chunks)} chunks")

        if self.vector_store is not None:
            self.vector_store.add_documents(chunks)
        else:
            # Vectors are normalized (see embeddings config), so MAX_INNER_PRODUCT
            # returns cosine similarity directly: score in 0-1, higher = better.
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_store.save_local(self.persist_dir)
        print(f"Stored {len(chunks)} chunks in '{self.collection_name}' index")
        return len(chunks)

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """
        Return top-k chunks above the similarity threshold.
        With normalized embeddings + MAX_INNER_PRODUCT, similarity_search_with_score
        returns the cosine similarity directly (0-1, higher = more similar).
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results if score >= SIMILARITY_THRESHOLD]

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None
