"""
FastAPI server for the Corporate RAG Chatbot.

Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

try:
    from main import CorporateChatbot
    from config import API_HOST, API_PORT
except ImportError:
    from prototype.main import CorporateChatbot
    from prototype.config import API_HOST, API_PORT

STATIC_DIR = Path(__file__).parent / "static"
FEEDBACK_LOG = Path(__file__).parent / "feedback_log.jsonl"


# --- Request / Response models ---

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    # In production this would come from the JWT; here it's a demo field
    department: Optional[str] = None
    # Override the global FAITHFULNESS_CHECK config per request (e.g. high-stakes query)
    check_faithfulness: Optional[bool] = None


class QueryResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float
    confidence_level: str  # "high" | "medium" | "low" | "out_of_scope" | "unavailable"
    session_id: str
    faithfulness: Optional[float] = None      # 0-1 groundedness score (if checked)
    faithfulness_warning: bool = False        # true if below threshold


class DocumentInput(BaseModel):
    content: str
    source: str
    category: str = "General"
    department: str = "all"


class IngestRequest(BaseModel):
    documents: List[DocumentInput]


class IngestResponse(BaseModel):
    chunks_added: int
    documents_processed: int


class HealthResponse(BaseModel):
    status: str
    knowledge_base_ready: bool
    sessions_active: int


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: str  # "up" | "down"
    session_id: str = "default"
    comment: Optional[str] = None


# --- App lifecycle ---

# One chatbot instance per session (keyed by session_id).
# Production would use Redis or a DB-backed store.
sessions: dict[str, CorporateChatbot] = {}


def get_or_create_session(session_id: str) -> CorporateChatbot:
    if session_id not in sessions:
        bot = CorporateChatbot(collection_name="default")
        sessions[session_id] = bot
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the default session with mock data on startup
    default_bot = CorporateChatbot(collection_name="default")
    default_bot.load_documents_from_directory()
    sessions["default"] = default_bot
    print("Default session initialized.")
    yield
    sessions.clear()


app = FastAPI(
    title="Corporate RAG Chatbot",
    description="Internal knowledge base chatbot — prototype",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/")
def root():
    """Serve the minimal chat UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Chat UI not found. API is running — see /docs."}


@app.get("/health", response_model=HealthResponse)
def health():
    default = sessions.get("default")
    return HealthResponse(
        status="ok",
        knowledge_base_ready=default.doc_processor.is_ready if default else False,
        sessions_active=len(sessions),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    bot = get_or_create_session(req.session_id)
    result = bot.query(req.query, check_faithfulness=req.check_faithfulness)

    return QueryResponse(
        response=result["response"],
        sources=result["sources"],
        confidence=result["confidence"],
        confidence_level=result["confidence_level"],
        session_id=req.session_id,
        faithfulness=result.get("faithfulness"),
        faithfulness_warning=result.get("faithfulness_warning", False),
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Server-Sent Events stream. Emits one `meta` event (sources + confidence),
    then a series of `token` events, then a `done` event. Lets the UI render
    the answer word-by-word and show sources before generation finishes.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    bot = get_or_create_session(req.session_id)

    def event_generator():
        for event in bot.query_stream(req.query, check_faithfulness=req.check_faithfulness):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Append a thumbs up/down rating to a JSONL log — the eval feedback loop."""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'.")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id,
        "query": req.query,
        "response": req.response,
        "rating": req.rating,
        "comment": req.comment,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"message": "Feedback recorded. Thank you."}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    from langchain.schema import Document

    docs = [
        Document(
            page_content=d.content,
            metadata={"source": d.source, "category": d.category, "department": d.department},
        )
        for d in req.documents
    ]

    # Use the default session's processor for ingestion
    bot = get_or_create_session("default")
    chunks_added = bot.doc_processor.ingest_documents(docs)

    return IngestResponse(
        chunks_added=chunks_added,
        documents_processed=len(docs),
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id].reset_history()
        return {"message": f"History cleared for session '{session_id}'"}
    raise HTTPException(status_code=404, detail="Session not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
