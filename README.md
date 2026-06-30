# Corporate RAG Chatbot

A prototype internal knowledge-base assistant that answers employee questions by retrieving
from HR policies, technical documentation, and project wikis — grounded in real documents,
not LLM hallucination.

> **Scope note**: This is a working proof-of-concept, not a production system. Where production
> concerns (auth, hybrid retrieval, re-ranking, eval harness) are not implemented, they are
> explicitly documented in [ARCHITECTURE.md](ARCHITECTURE.md) and called out in the code.
> A thoughtful incomplete design beats a shallow "complete" one.

## TL;DR

- **What it is:** RAG chatbot over internal docs (HR / technical / project wiki), with source citations and out-of-scope refusal.
- **Stack:** FastAPI · LangChain · FAISS · local MiniLM embeddings · GPT-4o. Vanilla-JS streaming chat UI.
- **Works today:** ingestion → chunk → embed → retrieve → grounded answer with citations; streaming, conversation memory, thumbs up/down feedback. Verified end-to-end.
- **Deliberately deferred (designed, not built):** JWT auth + role-based access, hybrid/re-rank retrieval, eval harness — see [ARCHITECTURE.md](ARCHITECTURE.md).
- **Cost:** ~$278/month for 1,000 employees (tiered models + caching, self-hosted vector DB); the 3-month agentic target is ~$615/month. See [COST_MODEL.md](costs/COST_MODEL.md).
- **Run it:** add a key to `prototype/.env`, then `uvicorn api:app --port 8000` → open http://localhost:8000.

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
VECTOR_DB_PATH=./faiss_index
```

**Run the CLI demo** (ingests mock data, runs test queries):

```bash
cd prototype
python main.py
```

**Run the API server + chat UI**:

```bash
cd prototype
uvicorn api:app --reload --port 8000
```

- **Chat UI**: http://localhost:8000  — streaming answers, source citations, thumbs up/down
- **Swagger UI**: http://localhost:8000/docs

**Example API call** (non-streaming):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many PTO days do I get in my first year?", "session_id": "demo"}'
```

**Streaming** (Server-Sent Events):

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Who leads the Data & ML team?", "session_id": "demo"}'
```

> The UI, `/health`, `/feedback`, and out-of-scope queries work **without** an OpenAI key
> (the LLM is constructed lazily). Only actual answer generation needs `OPENAI_API_KEY`.
> Run `python test_retrieval.py` / `python test_api.py` for no-key smoke tests.

---

## Repository Structure

```
corporate-chatbot-rag/
├── prototype/
│   ├── main.py            # CorporateChatbot: retrieval, prompt, query + query_stream
│   ├── rag_pipeline.py    # DocumentProcessor: chunking, embedding, retrieval
│   ├── api.py             # FastAPI server (/query, /query/stream, /ingest, /feedback, /health)
│   ├── config.py          # All tuneable parameters
│   ├── requirements.txt
│   ├── test_retrieval.py  # No-key smoke test: ingestion + retrieval
│   ├── test_api.py        # No-key smoke test: API endpoints
│   ├── static/
│   │   └── index.html     # Minimal streaming chat UI (vanilla JS)
│   └── mock_data/         # Sample corporate documents (knowledge base)
│       ├── hr_policy.txt
│       ├── tech_docs.txt
│       └── project_wiki.txt
├── ARCHITECTURE.md        # System design + inline Mermaid diagrams (C4-style), all 6 required topics
├── .vscode/launch.json    # Debug configs (API server, CLI demo, tests)
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

### Vector DB: FAISS (local)

- Zero setup, persists to disk, runs in-process. Pure vector index — fast and dependency-light, with pre-built wheels on every platform (ChromaDB was the original choice but its native extension fails to build on Windows/Python 3.12 without C++ build tools; FAISS avoids that).
- Production path: **Weaviate** (self-hosted, strong metadata filtering, ~$0 beyond the host) or **Pinecone** (fully managed, but its Standard plan has a ~$50/month floor — usage at this scale is under $1, so you're paying the floor). The retrieval interface is the same — swapping is a config change.
- Key limitation FAISS does **not** solve, and why production needs a metadata-aware store: FAISS is a pure vector index with no native metadata filtering. The access-control and multi-tenancy design (see [ARCHITECTURE.md](ARCHITECTURE.md)) depends on server-side metadata filters at query time, which Weaviate/Pinecone/Chroma provide and FAISS does not.

### Orchestration: LangChain

- Provides the text splitter, vector store abstraction, and embedding wrappers that would otherwise need to be written from scratch.
- The tradeoff: LangChain adds abstraction weight and its API has changed significantly between 0.1 and 0.3. For a production system I'd evaluate whether a thin custom layer over the FAISS and OpenAI SDKs directly would be cleaner.
- Not using LangChain's chain abstractions (LCEL, ConversationChain): the prompt construction and history management in `main.py` is written explicitly, which makes it easier to reason about and debug.

### Document Parsing: PyPDF2 + plain text

- Sufficient for the mock data (`.txt`, `.md`).
- Production gap: PyPDF2 loses table structure in PDFs. **Unstructured.io** or **Azure Document Intelligence** handle tables, headers, and multi-column layouts correctly. This is a known limitation documented here.

### Backend: FastAPI

- Async, automatic OpenAPI docs at `/docs`, Pydantic validation on all request/response models.
- Chosen over Flask: FastAPI's type safety and auto-documentation reduce integration friction for whoever builds the frontend.

### Frontend

A minimal **single-file chat UI** ([prototype/static/index.html](prototype/static/index.html), vanilla JS, no build step) served at `localhost:8000`. It streams answers token-by-token over SSE, shows source citations and a confidence badge, and has thumbs up/down feedback. Kept deliberately framework-free so it's easy to read and runs with zero tooling.

For production: a React chat widget embedded in the company's intranet (Notion, Confluence, or custom) is the natural fit. The API contract is clean enough that a frontend engineer could build this independently.

### Observability

The prototype ships a basic **feedback loop**: `/feedback` logs thumbs up/down (with the query and response) to `feedback_log.jsonl` — the raw signal an eval pipeline would consume. Full tracing is not wired. Production stack:
- **LLM-observability platform — Phoenix (Arize)** or **Langfuse** (both open-source): capture **traces** that follow a single query through the whole pipeline (initial prompt → query sent to retriever → chunks returned → reranker → final prompt → generated response → latency at each step). This is what makes "where and why did this answer go wrong?" answerable, and where the RAGAS metrics above get logged per-trace. Also enables **A/B experiments** (does a new system prompt or a reranker actually improve quality?).
- **Classic infra monitoring — Datadog / Grafana / CloudWatch**: the things an LLM-observability tool doesn't cover — API latency, error rate, memory, vector DB query time.

---

## Implementation Decisions

### Retrieval bug fix

The original code filtered retrieved chunks with `score >= threshold`, but the vector store returns an L2 *distance* (lower = better), so the filter was keeping the *worst* matches and dropping the best ones. Fixed by making the score a genuine cosine similarity (0–1, higher = better):

1. Normalize embeddings at the model level (`encode_kwargs={"normalize_embeddings": True}`).
2. Build the FAISS index with `DistanceStrategy.MAX_INNER_PRODUCT` — on normalized vectors the inner product *is* the cosine similarity, so `similarity_search_with_score` returns an interpretable 0–1 score and `score >= threshold` now keeps the best matches.
3. Calibrated the threshold (0.35) empirically: genuine matches score ~0.36+, out-of-scope queries fall below ~0.31. See `test_retrieval.py` for the no-API-key smoke test that verifies this.

### Out-of-scope queries

If no chunk scores above the threshold (0.35), the system returns a "not in knowledge base" message **without calling the LLM**. This:
- Saves tokens on every unanswerable query.
- Prevents the LLM from hallucinating an answer from training data.
- Gives the user a useful redirect (contact HR, check Confluence).

### Faithfulness self-check (RAGAS-style groundedness)

An optional second LLM pass that verifies the generated answer is actually supported by the retrieved context — the output-side complement to the input-side out-of-scope gate. After generation, a cheap judge model (`gpt-4o-mini`, temperature 0) is asked to extract every factual claim and return `{"faithfulness": 0–1, "unsupported_claims": [...]}`. If the score falls below `FAITHFULNESS_THRESHOLD` (0.5), the response is flagged with a "please verify" warning and the unsupported claims are surfaced.

- **Off by default** (it adds one LLM call of latency + cost). Enable globally via `FAITHFULNESS_CHECK=true`, or per request: `query(..., check_faithfulness=True)` / `{"check_faithfulness": true}` on the API, or the **"Verify grounding"** checkbox in the chat UI.
- Implemented in `CorporateChatbot.check_faithfulness()`; verified both directions — grounded answers score 1.0, a deliberately fabricated answer ("unlimited remote work and a free company car") scores 0.0 and both invented claims are listed.
- This is the prototype version of the RAGAS `Faithfulness()` metric described in the [Evaluation section](#evaluation-approach); in production the same signal would be logged per-trace rather than blocking the response.

### Conversation memory

A simple rolling window of the last 3 turns (6 messages) is included in each prompt. This is enough for follow-up questions ("what about part-time employees?") without growing the prompt unboundedly. A production system would persist history to a DB keyed by session ID.

### Confidence levels

Each response includes a `confidence_level` field derived from the average cosine similarity of retrieved chunks:
- `high` (≥ 0.5): the question maps closely to the knowledge base.
- `medium` (0.42–0.5): relevant content found, but not an exact match.
- `low` (0.35–0.42): marginal match — surface this in the UI so the user knows to verify.
- `out_of_scope`: nothing above the 0.35 threshold.

> These bands are calibrated to the `all-MiniLM-L6-v2` cosine-score distribution on this corpus, where strong matches land ~0.5 and marginal ones ~0.36. A different embedding model would need re-calibration.

---

## Evaluation Approach

How do you know the chatbot is getting better (or worse)? We evaluate along two axes — **scope** (component vs. system) × **evaluator type** (code-based, LLM-as-judge, human feedback):

| | Code-based (cheap) | LLM-as-judge (flexible) | Human feedback (gold) |
|---|---|---|---|
| **Component** (retriever) | retrieval latency, tokens/query | retrieved-document relevance | annotated relevance labels |
| **System** (end-to-end) | answer latency, valid output | RAGAS faithfulness, citation accuracy | thumbs up/down |

### Component eval — the retriever

- **Context recall / precision**: build a golden dataset of 50–100 (question → expected source) pairs and check whether the correct chunk lands in top-k. Cheap, code-based, run on every PR.
- These directly target the "did we retrieve the right thing?" question before the LLM is even involved.

### System eval — the generated answer (RAGAS metrics)

Using the [RAGAS](https://docs.ragas.io) library, which the production-RAG canon standardizes on:
- **Faithfulness**: does every factual claim in the answer trace back to the retrieved context? (Catches hallucination — **implemented** in the prototype as an optional self-check; see [Faithfulness self-check](#faithfulness-self-check-ragas-style-groundedness).)
- **Response Relevancy**: is the answer actually addressing the question, regardless of correctness?
- **Citation Quality / Answer Correctness**: do cited sources support the claims; does the answer match the expected one?

All system-level metrics use **LLM-as-a-judge** under the hood (the LLM's role is too nuanced for pure code metrics) — with the known caveat that judges mildly favor their own model family.

### Online eval (production)

- **Thumbs up/down** on every response (already implemented via `/feedback`) — the human-feedback signal an eval pipeline consumes.
- **Out-of-scope rate**: a spike means a new topic isn't covered in the knowledge base.
- **Average retrieval score over time**: a drop signals corpus drift (new docs use different terminology).
- The flywheel: **observe traffic → evaluate performance → experiment with changes** (A/B test prompts, rerankers), then repeat.

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
- [x] FastAPI server with `/query`, `/query/stream`, `/ingest`, `/feedback`, `/health`
- [x] Mock data corpus covering HR, technical docs, project wiki
- [x] Streaming responses (SSE) + minimal chat UI
- [x] Thumbs up/down feedback loop (logged to JSONL)
- [x] RAGAS-style faithfulness self-check (optional, toggleable)
- [ ] Auth (designed in ARCHITECTURE.md, not implemented)
- [ ] Hybrid retrieval (designed as next step, not implemented)
- [ ] Role-based access enforcement (needs a metadata-aware vector store)

### 3-month production-ready build

**Agentic, multi-knowledge-base redesign** (the headline change — see [ARCHITECTURE.md §8](ARCHITECTURE.md), with container + agentic-core diagrams):
- Conversation agent + orchestrator agent, with **one specialized RAG ability per knowledge base** (HR / technical / wiki) plus non-RAG tools
- Typed Pydantic/JSON hand-offs (`Query` → `AbilityResult` → `Answer`) for formatting + anti-hallucination
- `UserContext` (role/department/purpose) for personalization and per-ability access scope
- Two-tier memory: last 5 messages verbatim + summarized long-term history

**Supporting production work:**
- Hybrid retrieval (BM25 + vector + cross-encoder re-ranking)
- JWT auth + metadata-filter access control per department
- Webhook-triggered incremental ingestion from Confluence / SharePoint / Google Drive
- Phoenix/Langfuse tracing; RAGAS eval harness with golden dataset, run on every PR
- Slack bot integration (employees ask questions where they already work)
- Semantic cache (Redis); model tiering; quantized embeddings
- PDF table extraction (Unstructured.io)
- Load testing and latency benchmarks; A/B experimentation; 10-user pilot rollout
