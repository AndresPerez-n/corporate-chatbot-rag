"""
Corporate RAG Chatbot — prototype entry point.

Run from within the prototype/ directory:
    python main.py

Or from the project root:
    python -m prototype.main
"""

from pathlib import Path
from typing import List, Iterator, Optional, Tuple

from langchain.schema import Document

try:
    from config import LLM_MODEL, OPENAI_API_KEY
    from rag_pipeline import DocumentProcessor
except ImportError:
    from prototype.config import LLM_MODEL, OPENAI_API_KEY
    from prototype.rag_pipeline import DocumentProcessor

from langchain_openai import ChatOpenAI


MOCK_DATA_DIR = Path(__file__).parent / "mock_data"

SYSTEM_PROMPT = """You are a helpful internal assistant for Acme Corp employees.
Answer questions using ONLY the context provided below. Each context block is tagged
with its source document.

If the answer is not in the context, say exactly:
"I don't have information about that in the knowledge base. Try asking HR (hr@acmecorp.com) or checking Confluence."

Never make up policies, numbers, or names. If you're uncertain, say so.
Always cite the source document at the end of your answer."""


class CorporateChatbot:
    """Main chatbot: ingests docs, retrieves context, generates grounded answers."""

    def __init__(self, collection_name: str = "default"):
        self.doc_processor = DocumentProcessor(collection_name=collection_name)
        # LLM is built lazily on first use: the UI, health, feedback, and
        # out-of-scope paths never call it, so the app should run without a key.
        self._llm: Optional[ChatOpenAI] = None
        # Simple in-memory conversation history (last 3 turns = 6 messages)
        self.conversation_history: List[dict] = []
        self.MAX_HISTORY_TURNS = 3

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            if not OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to prototype/.env to enable "
                    "answer generation. (Retrieval, the UI, and feedback work without it.)"
                )
            self._llm = ChatOpenAI(
                model_name=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.2,  # low temp keeps answers factual and reproducible
            )
        return self._llm

    def load_documents_from_directory(self, directory: Path = MOCK_DATA_DIR) -> None:
        """Load all .txt and .md files from a directory into the vector store."""
        docs = []
        for filepath in directory.glob("*"):
            if filepath.suffix in (".txt", ".md"):
                content = filepath.read_text(encoding="utf-8")
                category = self._infer_category(filepath.name)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": filepath.name,
                        "category": category,
                        "department": "all",
                    },
                ))
        if not docs:
            print(f"No documents found in {directory}")
            return
        print(f"Loading {len(docs)} documents from {directory}...")
        self.doc_processor.ingest_documents(docs)
        print("Knowledge base ready.")

    def _infer_category(self, filename: str) -> str:
        if "hr" in filename.lower() or "policy" in filename.lower():
            return "HR"
        if "tech" in filename.lower() or "doc" in filename.lower():
            return "Technical"
        if "wiki" in filename.lower() or "project" in filename.lower():
            return "Wiki"
        return "General"

    def _prepare(self, user_query: str) -> Tuple[Optional[str], dict]:
        """
        Retrieve context and build the prompt. Shared by query() and query_stream().

        Returns (prompt, meta). If prompt is None the request should short-circuit
        and meta already holds the full canned response (not-ready / out-of-scope).
        Otherwise meta holds sources/confidence/confidence_level for the answer.
        """
        if not self.doc_processor.is_ready:
            return None, {
                "response": "The knowledge base is not initialized. Please load documents first.",
                "sources": [],
                "confidence": 0.0,
                "confidence_level": "unavailable",
            }

        retrieved = self.doc_processor.retrieve(user_query)

        # Out-of-scope: nothing retrieved above threshold
        if not retrieved:
            return None, {
                "response": (
                    "I couldn't find relevant information in the knowledge base for that question. "
                    "Try rephrasing, or contact hr@acmecorp.com / #it-help for direct support."
                ),
                "sources": [],
                "confidence": 0.0,
                "confidence_level": "out_of_scope",
            }

        # Build context block with source attribution per chunk
        context_blocks = []
        for doc, _ in retrieved:
            source = doc.metadata.get("source", "unknown")
            context_blocks.append(f"[Source: {source}]\n{doc.page_content.strip()}")
        context = "\n\n---\n\n".join(context_blocks)

        # Build history string (last N turns)
        history_str = ""
        if self.conversation_history:
            history_str = "\n\nPrevious conversation:\n"
            for msg in self.conversation_history[-(self.MAX_HISTORY_TURNS * 2):]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}
{history_str}
Current question: {user_query}"""

        # Score bands calibrated to the all-MiniLM-L6-v2 cosine distribution,
        # where strong matches land ~0.5 and marginal ones ~0.36.
        avg_score = sum(s for _, s in retrieved) / len(retrieved)
        confidence_level = (
            "high" if avg_score >= 0.5
            else "medium" if avg_score >= 0.42
            else "low"
        )

        meta = {
            "sources": list({doc.metadata.get("source", "unknown") for doc, _ in retrieved}),
            "confidence": round(avg_score, 3),
            "confidence_level": confidence_level,
        }
        return prompt, meta

    def _record_turn(self, user_query: str, answer: str) -> None:
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": answer})

    def query(self, user_query: str) -> dict:
        """Process a user query. Returns response, sources, and confidence info."""
        prompt, meta = self._prepare(user_query)
        if prompt is None:
            return meta

        answer = self.llm.invoke(prompt).content
        self._record_turn(user_query, answer)
        return {"response": answer, **meta}

    def query_stream(self, user_query: str) -> Iterator[dict]:
        """
        Streaming variant. Yields a sequence of events:
          {"type": "meta",  ...sources/confidence...}   (always first)
          {"type": "token", "content": "..."}           (zero or more)
          {"type": "done"}                               (always last)
        For short-circuit cases the canned response is emitted as a single token.
        """
        prompt, meta = self._prepare(user_query)

        if prompt is None:
            canned = meta.pop("response")
            yield {"type": "meta", **meta}
            yield {"type": "token", "content": canned}
            yield {"type": "done"}
            return

        yield {"type": "meta", **meta}
        parts: List[str] = []
        for chunk in self.llm.stream(prompt):
            token = chunk.content
            if token:
                parts.append(token)
                yield {"type": "token", "content": token}

        self._record_turn(user_query, "".join(parts))
        yield {"type": "done"}

    def reset_history(self) -> None:
        self.conversation_history = []


if __name__ == "__main__":
    chatbot = CorporateChatbot()
    chatbot.load_documents_from_directory()

    test_queries = [
        "What is the work from home policy?",
        "How many PTO days do I get in my first year?",
        "How do I roll back a database migration?",
        "Who is the lead of the Data & ML team?",
        "What's the lunch subsidy policy?",  # intentionally out of scope
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)
        result = chatbot.query(q)
        print(f"Response: {result['response']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']} ({result['confidence_level']})")
