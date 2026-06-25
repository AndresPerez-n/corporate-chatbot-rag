# Corporate RAG Chatbot

A prototype internal knowledge-base assistant that answers employee questions by retrieving
from HR policies, technical documentation, and project wikis — grounded in real documents,
not LLM hallucination.

> **Scope note**: This is a working proof-of-concept, not a production system. Where production
> concerns (auth, hybrid retrieval, re-ranking, eval harness) are not implemented, they are
> explicitly documented in [ARCHITECTURE.md](ARCHITECTURE.md) and called out in the code.
> A thoughtful incomplete design beats a shallow "complete" one.

---

## Quick Start

```bash
cd prototype
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create `.env` in the project root (or in `prototype/`):

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o          # optional override
VECTOR_DB_PATH=./chromadb_data
```

**Run the CLI demo** (ingests mock data, runs test queries):

```bash
cd prototype
python main.py
```

**Run the API server**:

```bash
cd prototype
uvicorn api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

**Example API call**:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many PTO days do I get in my first year?", "session_id": "demo"}'
```

---

## Repository Structure

```
corporate-chatbot-rag/
├── prototype/
│   ├── main.py            # CorporateChatbot class + CLI demo
│   ├── rag_pipeline.py    # DocumentProcessor: chunking, embedding, retrieval
│   ├── api.py             # FastAPI server (/query, /ingest, /health)
│   ├── config.py          # All tuneable parameters
│   ├── requirements.txt
│   └── mock_data/         # Sample corporate documents (knowledge base)
│       ├── hr_policy.txt
│       ├── tech_docs.txt
│       └── project_wiki.txt
├── ARCHITECTURE.md        # System design, diagrams, all 6 required topics
└── costs/
    └── COST_MODEL.md      # Cost breakdown with assumptions and math
```

---

## Tech Stack — With Rationale

### LLM: GPT-4o (OpenAI)

Chosen over alternatives for:
- **GPT-4o vs. Claude Sonnet**: both are strong. GPT-4o has slightly better structured-output reliability for this use case; Claude Sonnet has better instruction-following for safety-sensitive domains. In a real engagement I'd run evals on both before committing.
- **GPT-4o vs. GPT-4o-mini**: mini is used for simple queries under model tiering (see cost model). Complex multi-document reasoning stays on GPT-4o.
- **Not self-hosted**: running Llama-3 locally would eliminate per-token cost but adds GPU infra (~$200–500/month for a decent inference server) and maintenance burden. Not worth it at 1,000-employee scale unless data residency requirements demand it.

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

- Runs **locally on CPU** — zero per-query API cost and no external dependency for embeddings.
- 384-dimensional vectors: small enough for fast ANN search, large enough for good semantic coverage on corporate English text.
- Chosen over `text-embedding-3-small` (OpenAI): the OpenAI model is marginally better on benchmarks but adds latency, API cost, and a runtime dependency. For a prototype, local wins.

### Vector DB: ChromaDB (local)

- Zero setup, persists to disk, runs in-process. Ideal for a prototype.
- Production path: **Weaviate** (self-hosted, strong metadata filtering) or **Pinecone** (fully managed, ~$5/month at this scale). The retrieval interface is the same — swapping is a config change.
- Key production requirement not met by ChromaDB: no built-in auth or multi-tenant isolation beyond metadata filters.

### Orchestration: LangChain

- Provides the text splitter, vector store abstraction, and embedding wrappers that would otherwise need to be written from scratch.
- The tradeoff: LangChain adds abstraction weight and its API has changed significantly between 0.1 and 0.3. For a production system I'd evaluate whether a thin custom layer over the Chroma and OpenAI SDKs directly would be cleaner.
- Not using LangChain's chain abstractions (LCEL, ConversationChain): the prompt construction and history management in `main.py` is written explicitly, which makes it easier to reason about and debug.

### Document Parsing: PyPDF2 + plain text

- Sufficient for the mock data (`.txt`, `.md`).
- Production gap: PyPDF2 loses table structure in PDFs. **Unstructured.io** or **Azure Document Intelligence** handle tables, headers, and multi-column layouts correctly. This is a known limitation documented here.

### Backend: FastAPI

- Async, automatic OpenAPI docs at `/docs`, Pydantic validation on all request/response models.
- Chosen over Flask: FastAPI's type safety and auto-documentation reduce integration friction for whoever builds the frontend.

### Frontend

Not implemented in this prototype. The `/docs` Swagger UI at `localhost:8000/docs` serves as the interactive interface for demo purposes.

For production: a React chat widget embedded in the company's intranet (Notion, Confluence, or custom) is the natural fit. The API contract is clean enough that a frontend engineer could build this independently.

### Observability

Not wired in the prototype. Production stack:
- **Langfuse** (open-source): traces every LLM call with latency, token count, and cost. Provides a feedback loop (thumbs up/down per response).
- **Datadog / CloudWatch**: infra metrics (API latency, error rate, vector DB query time).

---

## Implementation Decisions

### Retrieval bug fix

The original code used `similarity_search_with_score` with a filter `score >= threshold`. Chroma's default distance metric is L2 (lower = better), so that filter was keeping the *worst* matches. Fixed by:

1. Setting `collection_metadata={"hnsw:space": "cosine"}` — cosine similarity, 0–1, higher is better.
2. Using `similarity_search_with_relevance_scores` which normalises scores to 0–1 regardless of the underlying distance metric.

### Out-of-scope queries

If no chunk scores above the threshold (0.4), the system returns a "not in knowledge base" message **without calling the LLM**. This:
- Saves tokens on every unanswerable query.
- Prevents the LLM from hallucinating an answer from training data.
- Gives the user a useful redirect (contact HR, check Confluence).

### Conversation memory

A simple rolling window of the last 3 turns (6 messages) is included in each prompt. This is enough for follow-up questions ("what about part-time employees?") without growing the prompt unboundedly. A production system would persist history to a DB keyed by session ID.

### Confidence levels

Each response includes a `confidence_level` field derived from the average cosine similarity of retrieved chunks:
- `high` (≥ 0.7): the question maps closely to the knowledge base.
- `medium` (0.55–0.7): relevant content found, but not an exact match.
- `low` (0.4–0.55): marginal match — surface this in the UI so the user knows to verify.
- `out_of_scope`: nothing above threshold.

---

## Evaluation Approach

How do you know the chatbot is getting better (or worse)?

**Offline eval (before any change ships)**

Build a golden dataset: 50–100 (question, expected answer, source document) triples derived from the actual knowledge base. Run the RAG pipeline on each question and score:
- **Retrieval recall**: did the correct source document appear in the top-k?
- **Answer faithfulness**: does the generated answer contradict the retrieved context? (Use an LLM-as-judge prompt or a tool like Ragas.)
- **Answer correctness**: does it match the expected answer? (ROUGE-L for factual questions, LLM judge for open-ended.)

**Online eval (in production)**

- Thumbs up/down on every response, surfaced in Langfuse.
- Track the fraction of queries that hit `out_of_scope` — a sudden spike means a new topic isn't covered in the knowledge base.
- Monitor average retrieval score over time — a drop often means the corpus has drifted (new documents use different terminology).

---

## Key Risks and Open Questions

| Risk | Severity | Mitigation |
|---|---|---|
| LLM refuses to answer from context alone (over-refusal) | Medium | Tune system prompt; eval on adversarial examples |
| PDF table extraction loses structured data | High | Replace PyPDF2 with Unstructured.io before ingesting real docs |
| Stale knowledge base (docs updated but not re-ingested) | High | Webhook-triggered re-ingestion on document save events |
| Prompt injection via malicious document content | Medium | Sanitise chunk content before inserting into prompt; rate-limit by user |
| Embedding model drift (switching models invalidates all vectors) | Low | Document the model name + version in collection metadata; re-index on model change |
| High query volume from power users blowing past cost estimates | Low | Per-user rate limiting at the API layer |

---

## 4-Week vs. 3-Month Build

### 4-week MVP (what this prototype is)

- [x] RAG pipeline: chunking, embedding, cosine retrieval
- [x] Out-of-scope detection
- [x] Source citation in responses
- [x] Conversation memory (rolling window)
- [x] FastAPI server with `/query`, `/ingest`, `/health`
- [x] Mock data corpus covering HR, technical docs, project wiki
- [ ] Auth (describe design, don't implement)
- [ ] Hybrid retrieval (document as next step)
- [ ] Frontend (Swagger UI is sufficient for demo)

### 3-month production-ready build

- Hybrid retrieval (BM25 + vector + cross-encoder re-ranking)
- JWT auth + metadata-filter access control per department
- Webhook-triggered incremental ingestion from Confluence / SharePoint / Google Drive
- Langfuse tracing + thumbs-up/down feedback loop
- Eval harness with golden dataset, run on every PR
- Slack bot integration (employees ask questions where they already work)
- Semantic cache (Redis) for repeated FAQ queries
- Role-based document access levels
- PDF table extraction (Unstructured.io)
- Load testing and latency benchmarks
