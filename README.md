# Corporate RAG Chatbot

A prototype internal knowledge-base assistant that answers employee questions by retrieving
from HR policies, technical documentation, and project wikis — grounded in source documents
rather than model recall.

> **Scope note.** This is a working proof-of-concept, not a production system. Production
> concerns that are not implemented (authentication, hybrid retrieval, re-ranking, evaluation
> harness) are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and annotated in the code.
> Scope has been reduced deliberately where noted; each omission is documented rather than
> hidden.

## Summary

- **Purpose:** a RAG assistant over internal knowledge bases (HR, technical, project wiki), with source citations and out-of-scope refusal.
- **Stack:** FastAPI · LangChain · FAISS · local MiniLM embeddings · GPT-4o; a dependency-free JavaScript streaming chat interface.
- **Implemented today:** ingestion → chunking → embedding → retrieval → grounded answer with citations; response streaming, conversation memory, and a thumbs up/down feedback loop. Verified end-to-end.
- **Deliberately deferred (designed, not built):** JWT authentication with role-based access, hybrid and re-ranked retrieval, and an evaluation harness — see [ARCHITECTURE.md](ARCHITECTURE.md).
- **Cost:** approximately $278/month for 1,000 employees (tiered models with caching, self-hosted vector store); the 3-month agentic target is approximately $615/month. See [COST_MODEL.md](costs/COST_MODEL.md).
- **To run:** provide an API key in `prototype/.env`, then `uvicorn api:app --port 8000` and open http://localhost:8000.

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

**Run the CLI demonstration** (ingests mock data, runs sample queries):

```bash
cd prototype
python main.py
```

**Run the API server and chat interface:**

```bash
cd prototype
uvicorn api:app --reload --port 8000
```

- **Chat interface:** http://localhost:8000 — streaming answers, source citations, thumbs up/down
- **API documentation (Swagger):** http://localhost:8000/docs

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

> The interface, `/health`, `/feedback`, and out-of-scope queries operate **without** an OpenAI
> key (the LLM is instantiated lazily). Only answer generation requires `OPENAI_API_KEY`.
> Run `python test_retrieval.py` and `python test_api.py` for the no-key smoke tests.

---

## Repository Structure

```
corporate-chatbot-rag/
├── prototype/
│   ├── main.py            # CorporateChatbot: retrieval, prompt, query + query_stream
│   ├── rag_pipeline.py    # DocumentProcessor: chunking, embedding, retrieval
│   ├── api.py             # FastAPI server (/query, /query/stream, /ingest, /feedback, /health)
│   ├── config.py          # Tunable parameters
│   ├── requirements.txt
│   ├── test_retrieval.py  # No-key smoke test: ingestion + retrieval
│   ├── test_api.py        # No-key smoke test: API endpoints
│   ├── static/
│   │   └── index.html     # Streaming chat interface (dependency-free JavaScript)
│   └── mock_data/         # Sample corporate documents (knowledge base)
│       ├── hr_policy.txt
│       ├── tech_docs.txt
│       └── project_wiki.txt
├── ARCHITECTURE.md        # System design + inline Mermaid diagrams (C4-style), all six required topics
├── .vscode/launch.json    # Debug configurations (API server, CLI demo, tests)
└── costs/
    └── COST_MODEL.md      # Cost breakdown with assumptions and calculations
```

---

## Tech Stack — With Rationale

### LLM: GPT-4o (OpenAI)

Selected over the alternatives considered:
- **GPT-4o vs. Claude Sonnet:** both are strong. GPT-4o offers marginally better structured-output reliability for this use case; Claude Sonnet offers stronger instruction-following for safety-sensitive domains. A production engagement would benchmark both before committing.
- **GPT-4o vs. GPT-4o-mini:** the mini model serves simple queries under model tiering (see the cost model). Complex multi-document reasoning remains on GPT-4o.
- **Not self-hosted:** running Llama-3 locally would remove per-token cost but introduces GPU infrastructure (approximately $200–500/month for a capable inference server) and maintenance overhead. Not justified at 1,000-employee scale unless data-residency requirements demand it.
- **Model choice is configuration, not lock-in:** `LLM_MODEL` is environment-driven. Newer frontier models are now less expensive than GPT-4o (GPT-4.1, GPT-5, and Claude Sonnet/Haiku — see the [COST_MODEL pricing references](costs/COST_MODEL.md)); the design supports moving to whichever model performs best on the evaluation set without code changes. GPT-4o is retained as a conservative, well-understood baseline.
- **GCP-native option — Gemini on Vertex AI:** because the organization operates on GCP, the principal advantage of **Gemini 2.x** (via Vertex AI) is consolidation: the system stays within a **single GCP project**, with unified IAM and billing, VPC Service Controls for data residency, and no data leaving Google's network — directly relevant to the requirement to keep internal documents private. Recommendation: use GPT-4o for the prototype; if GCP consolidation is a priority, Gemini on Vertex is the natural substitution (the `langchain-google-vertexai` integration makes this a configuration change rather than a rewrite).

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

- Runs **locally on CPU** — no per-query API cost and no external dependency for embeddings.
- 384-dimensional vectors: small enough for fast approximate-nearest-neighbour search, large enough for good semantic coverage of corporate English text.
- Selected over `text-embedding-3-small` (OpenAI): the OpenAI model is marginally stronger on benchmarks but adds latency, API cost, and a runtime dependency. For a prototype, the local model is the preferable tradeoff.

### Vector store: FAISS (local)

- Zero setup, persists to disk, runs in-process. A pure vector index — fast and dependency-light, with pre-built wheels on every platform. (ChromaDB was the original selection, but its native extension fails to build on Windows/Python 3.12 without C++ build tools; FAISS avoids that constraint.)
- Production path: **Weaviate** (self-hosted, strong metadata filtering, approximately $0 beyond the host) or **Pinecone** (fully managed; note that its Standard plan carries an approximately $50/month floor, and usage at this scale is under $1/month, so the plan floor — not usage — drives the cost). The retrieval interface is identical, so substitution is a configuration change.
- The limitation FAISS does **not** address, and why production requires a metadata-aware store: FAISS is a pure vector index with no native metadata filtering. The access-control and multi-tenancy design (see [ARCHITECTURE.md](ARCHITECTURE.md)) depends on server-side metadata filters at query time, which Weaviate, Pinecone, and Chroma provide and FAISS does not.

### Orchestration: LangChain

- Provides the text splitter, vector-store abstraction, and embedding wrappers that would otherwise require custom implementation.
- **vs. LlamaIndex:** LlamaIndex is more opinionated and retrieval-first, which is convenient initially but harder to adapt when custom agent/orchestration logic is required. LangChain's looser primitives suit the planned agentic redesign better.
- **vs. custom:** LangChain adds abstraction weight, and its API changes between minor versions. For production, a thin custom layer over the FAISS and OpenAI SDKs would be evaluated. The implementation already avoids LangChain's chain abstractions (LCEL, ConversationChain): prompt construction and history management in `main.py` are written explicitly, which improves readability and debuggability.

### Document Parsing & Chunking

**Parsing** — PyPDF2 with plain text:
- Sufficient for the mock data (`.txt`, `.md`).
- Production gap: PyPDF2 loses table structure in PDFs. **Unstructured.io** or **Azure Document Intelligence** handle tables, headers, and multi-column layouts correctly. This is a documented limitation.

**Chunking** — `RecursiveCharacterTextSplitter`, 512 characters / 64 overlap:
- 512 characters (approximately 128 tokens) balances precision (small enough that a chunk covers one topic) against coherence (large enough to contain a complete policy rule); the 64-character overlap prevents splitting a fact across a boundary.
- Selected over fixed-size or token splitters: recursive splitting respects paragraph and sentence boundaries first, keeping chunks semantically clean. Full rationale and the per-domain 3-month strategy are in [ARCHITECTURE.md §2 and §8](ARCHITECTURE.md).

### Backend & Hosting: FastAPI

- **Framework — FastAPI** (vs. Flask): asynchronous, automatic OpenAPI documentation at `/docs`, and Pydantic validation on every request/response model. The type safety and generated documentation reduce integration effort for the frontend.
- **Hosting/infrastructure:** the application is a stateless container. **GCP Cloud Run** (or AWS Fargate) is the default: scale-to-zero keeps idle cost near zero and autoscaling absorbs bursts, which suits variable internal traffic. A standard **EC2 t3.medium** is the simpler always-on alternative (approximately $30/month) where cold starts are undesirable. The managed vector store, Redis cache, and metrics database are separate services (see [COST_MODEL.md](costs/COST_MODEL.md)).

### Frontend

A single-file chat interface ([prototype/static/index.html](prototype/static/index.html), dependency-free JavaScript, no build step) served at `localhost:8000`. It streams answers token-by-token over SSE, displays source citations and a confidence indicator, and provides thumbs up/down feedback. It is intentionally framework-free for readability and zero build tooling.

For production, a React chat widget embedded in the corporate intranet (Notion, Confluence, or a custom portal) is the recommended approach. The API contract is well-defined enough for a frontend engineer to build against independently.

### Observability, Logging & Evaluation Tooling

The prototype includes a basic **feedback loop**: `/feedback` writes thumbs up/down ratings (with the query and response) to `feedback_log.jsonl` — the raw signal an evaluation pipeline consumes. Full tracing is not wired. Production stack:
- **LLM observability / tracing — Langfuse** (open-source): captures **traces** that follow a single query through the entire pipeline (prompt → retriever → chunks → reranker → final prompt → response → per-step latency), enabling root-cause analysis of individual answers. Selected over **Phoenix (Arize)** — both are strong and support LLM spans natively, but Langfuse's self-hostable server, prompt management, and native feedback/evaluation scoring (it ingests the thumbs up/down and RAGAS scores directly) best fit the feedback-loop design. Selected over a custom logging solution because LLM span tracing does not warrant reimplementation.
- **Evaluation tooling — RAGAS:** faithfulness, response relevancy, context precision/recall, and citation quality — run in CI against a golden Q&A set and logged per trace (see [Evaluation Approach](#evaluation-approach)). The prototype already implements a RAGAS-style faithfulness check in code.
- **Logging / infrastructure monitoring — Cloud Logging and Cloud Monitoring** (the organization operates on GCP; Datadog and Grafana are vendor-neutral equivalents): API latency, error rate, memory, and vector-store query time.
- **Experimentation — A/B testing** of prompts and rerankers, with results recorded in the metrics database.

### Deploying on GCP (the organization's cloud)

The stack is cloud-agnostic, but because the organization operates on GCP, the GCP-native mapping is below. The advantage is consolidation into a **single project**: unified IAM and billing, VPC Service Controls for data residency, and internal data that never leaves Google's network.

| Layer | Prototype / default | GCP-native equivalent |
|---|---|---|
| LLM | GPT-4o (OpenAI) | **Gemini 2.x on Vertex AI** |
| Embeddings | local MiniLM | Vertex AI text embeddings (or retain MiniLM on Cloud Run) |
| Vector store | FAISS | **Vertex AI Vector Search**, or **AlloyDB / Cloud SQL `pgvector`** (metadata filtering for access control) |
| Document parsing | PyPDF2 | **Google Document AI** (tables, OCR, layout) |
| Backend/hosting | FastAPI on Cloud Run | **Cloud Run** (already the selection) with **Artifact Registry** |
| Files / raw documents | object storage | **Cloud Storage (GCS)** |
| Cache / sessions | Redis | **Memorystore (Redis)** |
| Metrics database | Postgres | **Cloud SQL / AlloyDB** |
| Secrets | `.env` | **Secret Manager** |
| Observability | Langfuse | Langfuse on Cloud Run/GKE with **Cloud Logging/Monitoring** |
| Authentication perimeter | (none, local) | **Identity-Aware Proxy (IAP)** with Google SSO |

Recommendation: retain OpenAI for the prototype (fastest path to a working demonstration); if the organization requires consolidation in GCP, the substitutions above — most importantly **Gemini on Vertex** — keep data and billing within a single project without altering the architecture.

---

## Implementation Decisions

### Prompt design

The prompt ([`main.py`](prototype/main.py), `SYSTEM_PROMPT`) is constructed to keep answers grounded and verifiable:
- **Structured template:** system instructions → retrieved context (each block tagged `[Source: filename]`) → conversation history → current question. The consistent order ensures the model can always distinguish context from question.
- **Grounding instruction:** answer using only the provided context; if the answer is not present, return a fixed refusal string. This converts retrieval misses into an explicit "not found" response rather than a hallucinated answer.
- **Citation instruction:** cite the source document, making every claim verifiable and feeding the faithfulness check.
- **Low temperature (0.2):** favours factual, reproducible answers over creative phrasing.
- **Per-source tagging** allows both the model and the user to attribute each fact to a document. Full prompt structure and context-management rationale are in [ARCHITECTURE.md §4](ARCHITECTURE.md).

### Retrieval correction

The original implementation filtered retrieved chunks with `score >= threshold`, but the vector store returned an L2 *distance* (lower is better), so the filter retained the *worst* matches and discarded the best. This was corrected by making the score a true cosine similarity (0–1, higher is better):

1. Normalize embeddings at the model level (`encode_kwargs={"normalize_embeddings": True}`).
2. Build the FAISS index with `DistanceStrategy.MAX_INNER_PRODUCT` — on normalized vectors the inner product is the cosine similarity, so `similarity_search_with_score` returns an interpretable 0–1 score and `score >= threshold` retains the best matches.
3. Calibrated the threshold (0.35) empirically: genuine matches score approximately 0.36 and above, while out-of-scope queries fall below approximately 0.31. See `test_retrieval.py` for the no-key smoke test that verifies this.

### Ambiguous and out-of-scope queries

Two distinct failure modes, handled differently:

**Out-of-scope** (the question is not in the knowledge base). If no chunk scores above the threshold (0.35), the system returns a "not in knowledge base" message **without calling the LLM**. This:
- avoids token spend on unanswerable queries;
- prevents the model from generating an answer from training data;
- provides a useful redirect (contact HR, consult Confluence).

**Ambiguous / underspecified** (the question is in scope but vague, or matches weakly). Handled by degree rather than a hard gate:
- The `confidence_level` (below) is derived from the retrieval score, so a marginal match surfaces as **low confidence** in the interface — the user is prompted to verify rather than accept the answer uncritically.
- The grounding prompt instructs the model to state its uncertainty rather than guess, so a vague question yields a hedged, sourced answer rather than a confident but incorrect one.
- **3-month enhancement:** the conversation agent (ARCHITECTURE §8) can detect low-confidence or multi-intent queries and **ask a clarifying question** (for example, "do you mean the US or EU PTO policy?") before retrieving — the appropriate location for true disambiguation, which a single-shot prototype cannot perform cleanly.

### Faithfulness self-check (RAGAS-style groundedness)

An optional second LLM pass that verifies the generated answer is supported by the retrieved context — the output-side complement to the input-side out-of-scope gate. After generation, an inexpensive judge model (`gpt-4o-mini`, temperature 0) extracts every factual claim and returns `{"faithfulness": 0–1, "unsupported_claims": [...]}`. If the score falls below `FAITHFULNESS_THRESHOLD` (0.5), the response is flagged with a verification notice and the unsupported claims are surfaced.

- **Disabled by default** (it adds one LLM call of latency and cost). Enable globally via `FAITHFULNESS_CHECK=true`, or per request via `query(..., check_faithfulness=True)` / `{"check_faithfulness": true}` on the API, or the **"Verify grounding"** control in the chat interface.
- Implemented in `CorporateChatbot.check_faithfulness()` and verified in both directions: grounded answers score 1.0, while a deliberately fabricated answer ("unlimited remote work and a free company car") scores 0.0 with both invented claims listed.
- This is the prototype equivalent of the RAGAS `Faithfulness()` metric described in the [Evaluation Approach](#evaluation-approach); in production the same signal would be logged per trace rather than blocking the response.

### Conversation memory

A rolling window of the last three turns (six messages) is included in each prompt. This supports follow-up questions ("what about part-time employees?") without growing the prompt unbounded. A production system would persist history in a database keyed by session ID.

### Confidence levels

Each response includes a `confidence_level` field derived from the average cosine similarity of the retrieved chunks:
- `high` (≥ 0.5): the question maps closely to the knowledge base.
- `medium` (0.42–0.5): relevant content found, but not an exact match.
- `low` (0.35–0.42): marginal match — surfaced in the interface so the user verifies.
- `out_of_scope`: nothing above the 0.35 threshold.

> These bands are calibrated to the `all-MiniLM-L6-v2` cosine-score distribution on this corpus, where strong matches land near 0.5 and marginal ones near 0.36. A different embedding model would require re-calibration.

---

## Evaluation Approach

Answer quality is measured along two axes — **scope** (component vs. system) and **evaluator type** (code-based, LLM-as-judge, human feedback):

| | Code-based (low cost) | LLM-as-judge (flexible) | Human feedback (authoritative) |
|---|---|---|---|
| **Component** (retriever) | retrieval latency, tokens/query | retrieved-document relevance | annotated relevance labels |
| **System** (end-to-end) | answer latency, valid output | RAGAS faithfulness, citation accuracy | thumbs up/down |

### Component evaluation — the retriever

- **Context recall / precision:** build a golden dataset of 50–100 (question → expected source) pairs and verify that the correct chunk appears in the top-k. Low-cost, code-based, run on every pull request.
- These target retrieval correctness before the LLM is involved.

### System evaluation — the generated answer (RAGAS metrics)

Using the [RAGAS](https://docs.ragas.io) library, the standard in production RAG:
- **Faithfulness:** does every factual claim in the answer trace back to the retrieved context? (Detects hallucination — **implemented** in the prototype as an optional self-check; see [Faithfulness self-check](#faithfulness-self-check-ragas-style-groundedness).)
- **Response Relevancy:** does the answer address the question, independent of correctness?
- **Citation Quality / Answer Correctness:** do the cited sources support the claims, and does the answer match the expected one?

All system-level metrics rely on LLM-as-a-judge internally (the LLM's role is too nuanced for purely code-based metrics), with the known caveat that judges mildly favour their own model family.

### Online evaluation (production)

- **Thumbs up/down** on every response (implemented via `/feedback`) — the human-feedback signal an evaluation pipeline consumes.
- **Out-of-scope rate:** an increase indicates a topic not covered by the knowledge base.
- **Average retrieval score over time:** a decline indicates corpus drift (new documents using different terminology).
- The improvement cycle: **observe traffic → evaluate performance → experiment with changes** (A/B test prompts and rerankers), then repeat.

---

## Key Risks and Open Questions

| Risk | Severity | Mitigation |
|---|---|---|
| Over-refusal (model declines to answer from valid context) | Medium | Tune the system prompt; evaluate against adversarial examples |
| PDF table extraction loses structured data | High | Replace PyPDF2 with Unstructured.io before ingesting real documents |
| Stale knowledge base (documents updated but not re-ingested) | High | Webhook-triggered re-ingestion on document-save events |
| Prompt injection via malicious document content | Medium | Sanitize chunk content before prompt assembly; rate-limit per user |
| Embedding model drift (changing models invalidates all vectors) | Low | Record the model name and version in collection metadata; re-index on model change |
| Query volume from power users exceeding cost estimates | Low | Per-user rate limiting at the API layer |

---

## 4-Week vs. 3-Month Build

### 4-week MVP (the current prototype)

- [x] RAG pipeline: chunking, embedding, cosine retrieval
- [x] Out-of-scope detection
- [x] Source citation in responses
- [x] Conversation memory (rolling window)
- [x] FastAPI server with `/query`, `/query/stream`, `/ingest`, `/feedback`, `/health`
- [x] Mock data corpus covering HR, technical documentation, and project wiki
- [x] Streaming responses (SSE) with a chat interface
- [x] Thumbs up/down feedback loop (logged to JSONL)
- [x] RAGAS-style faithfulness self-check (optional, configurable)
- [ ] Authentication (designed in ARCHITECTURE.md, not implemented)
- [ ] Hybrid retrieval (designed as the next step, not implemented)
- [ ] Role-based access enforcement (requires a metadata-aware vector store)

### 3-month production build

**Agentic, multi-knowledge-base redesign** (the principal change — see [ARCHITECTURE.md §8](ARCHITECTURE.md), with container and agentic-core diagrams):
- Conversation agent and orchestrator agent, with **one specialized RAG ability per knowledge base** (HR / technical / wiki) plus non-RAG tools
- Typed Pydantic/JSON hand-offs (`Query` → `AbilityResult` → `Answer`) for formatting and hallucination control
- `UserContext` (role/department/purpose) for personalization and per-ability access scope
- Two-tier memory: the last five messages verbatim plus a summarized long-term history

**Supporting production work:**
- Hybrid retrieval (BM25 + vector + cross-encoder re-ranking)
- JWT authentication with metadata-filter access control per department
- Webhook-triggered incremental ingestion from Confluence / SharePoint / Google Drive
- Langfuse tracing with feedback/evaluation scoring; a RAGAS evaluation harness with a golden dataset, run on every pull request
- Slack integration (employees ask questions in their existing workflow)
- Semantic cache (Redis); model tiering; quantized embeddings
- PDF table extraction (Unstructured.io)
- Load testing and latency benchmarks; A/B experimentation; a 10-user pilot rollout
