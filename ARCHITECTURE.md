# Architecture Overview — Corporate RAG Chatbot

> **Rendered diagrams** (open in a browser): [`docs/architecture.html`](docs/architecture.html) — the prototype's current flow · [`docs/architecture-production.html`](docs/architecture-production.html) — the production "big solution". The Mermaid diagram below renders inline on GitHub.

## TL;DR

- **Query flow:** user → API (auth) → embed query → vector search → (re-rank) → inject top-k chunks into prompt → LLM → answer + citations.
- **Ingestion:** documents → parse → chunk (512 chars, 64 overlap) → embed (MiniLM) → store in vector index.
- **Retrieval:** dense vector search now; hybrid (BM25 + vector) + cross-encoder re-rank is the production design.
- **Access control:** enforced at the **retrieval layer** (metadata filter), never by asking the LLM to hide docs — that's not a security boundary.
- **Multi-tenancy:** single index + per-document `department`/`access_level` metadata filter; separate collections only for regulated isolation.
- **Prototype caveat:** FAISS (pure vector index) has no metadata filtering, so access control needs a metadata-aware store (Weaviate/Pinecone/Chroma) in production.

Each section below expands one of these points.

## System Diagram

```mermaid
flowchart TB
    subgraph ING["Ingestion Pipeline (offline / scheduled)"]
        SRC["Source Documents\n(PDF · MD · TXT)"] --> PARSE["Parser\n(PyPDF2 / plain text)"]
        PARSE --> CHUNK["Chunker\n(512 chars, 64 overlap)\nRecursiveCharacterSplitter"]
        CHUNK --> EMB1["Embedder\n(all-MiniLM-L6-v2)\n384-dim vectors, local"]
        EMB1 --> VDB[("Vector store\nFAISS (prototype) /\nWeaviate·Pinecone (prod)\ncosine index")]
    end

    subgraph QRY["Query Pipeline (real-time)"]
        USR["Employee"] --> UI["Chat UI / API\n(FastAPI)"]
        UI --> AUTH{{"JWT Auth\n+ ACL filter"}}
        AUTH --> EMB2["Embedder\n(same model)"]
        EMB2 --> VDB
        VDB --> RET["Top-k retrieval\n(cosine sim, k=5\nscore ≥ 0.35)"]
        RET --> RERANK["Re-ranker\n(cross-encoder)\n⚠ production only"]
        RERANK --> LLM["LLM\n(GPT-4o)"]
        LLM --> FAITH["Faithfulness\nself-check\n(optional)"]
        FAITH --> RESP["Response\n+ source citations\n+ confidence"]
        RESP --> USR
    end
```

---

## 1. End-to-End Data Flow

### Query path (real-time)

1. Employee submits a question through the chat UI or REST API.
2. FastAPI validates the JWT, extracts `{user_id, department, role}` from claims.
3. The query is embedded with the same model used at ingestion time (critical — model must match).
4. The vector store receives the query vector plus a metadata filter derived from the user's claims, ensuring they only see documents their role permits. (The prototype uses FAISS, a pure vector index; the metadata-filter step is a production capability provided by Weaviate/Pinecone/Chroma — see the note in section 6.)
5. Top-k chunks are retrieved by cosine similarity. In production a cross-encoder re-ranker would rescore these; the prototype skips re-ranking to reduce dependencies.
6. Retrieved chunks are injected into a structured prompt alongside the last 3 turns of conversation history.
7. The LLM generates a grounded response. System-prompt instructions prohibit it from going beyond the supplied context.
8. The response is returned with source citations so employees can verify the original document.

### Why this order matters

Retrieval happens before any LLM call — this is non-negotiable. Asking the LLM to "filter out" unauthorised content after retrieval is not a security boundary; the metadata filter at step 4 is.

---

## 2. Ingestion Pipeline

### Steps

| Step | Tool | Decision |
|---|---|---|
| Source crawl | Webhook / scheduled job | Triggered on document update, not polling |
| Parse | PyPDF2 (PDF), plain text (.txt/.md) | Handles the common formats; Unstructured.io would replace this in production for tables, embedded images |
| Chunk | `RecursiveCharacterTextSplitter` | 512 chars, 64 overlap |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` | Local, zero per-token cost |
| Store | FAISS (prototype) / Weaviate·Pinecone (prod), cosine index | See section 6 for namespace rationale |

### Chunking rationale

512 characters (≈ 128 tokens) is a deliberate middle ground:
- **Too small** (< 128 chars): individual chunks lose enough context that the LLM can't form a useful answer from them.
- **Too large** (> 1,024 chars): retrieval precision drops because each chunk covers multiple topics, so the similarity score gets diluted.

The 64-char overlap prevents a fact from being split across chunk boundaries (e.g., a policy rule that spans two paragraphs).

### What the prototype does NOT do (and why)

- **Incremental ingestion with change detection**: production would hash source documents and skip unchanged ones. Omitted here because mock data doesn't change.
- **Table extraction from PDFs**: PyPDF2 loses table structure. Unstructured.io or Azure Document Intelligence handle this correctly but add significant setup overhead for a prototype.
- **Chunk deduplication**: if the same document is ingested twice, chunks duplicate. A content-hash check before insert would prevent this.

---

## 3. Retrieval Strategy

### Prototype: dense vector search only

- Query embedded → cosine similarity against the FAISS index → top-5 chunks above threshold (0.35).
- Simple, easy to reason about, sufficient for a demo.

### Production: hybrid retrieval

Dense vector search alone has a known failure mode: it misses **exact keyword matches**. Ask "what is our ITAR compliance policy?" and the vector search may surface loosely related documents because "ITAR" is a rare acronym the embedding model hasn't seen in context.

The production design adds:

1. **BM25 keyword search** alongside vector search (sparse retrieval).
2. **Reciprocal Rank Fusion (RRF)** to merge the two ranked lists — no need to tune weights.
3. **Cross-encoder re-ranking** on the combined candidate set. A cross-encoder reads (query + chunk) together rather than comparing pre-computed vectors, which significantly improves precision at the cost of ~50–200ms latency.

The prototype explicitly skips steps 2 and 3. This is a conscious scope cut documented here rather than a gap left unaddressed.

---

## 4. LLM Orchestration

### Prompt structure

```
[System prompt — role, constraints, citation instruction]

Context:
[Source: hr_policy.txt]
<chunk text>
---
[Source: tech_docs.txt]
<chunk text>

Previous conversation:
User: ...
Assistant: ...

Current question: {user_query}
```

### Context management decisions

- **Max 5 chunks × 512 chars ≈ 2,560 chars of context** — stays well within any model's context window while keeping costs predictable.
- **Conversation history capped at 3 turns** (6 messages). Beyond that, older turns are dropped from the prompt. The alternative (full history) would grow the prompt unboundedly and inflate per-query token costs.
- **Temperature = 0.2** — low temperature for factual, consistent responses. We don't want creative paraphrasing of policy documents.

### Handling ambiguous or out-of-scope queries

Two layers of defence:

1. **Retrieval guard**: if no chunk scores above the similarity threshold, the system returns a canned "not in knowledge base" message without calling the LLM at all. This saves tokens and prevents hallucination.
2. **Prompt instruction**: the system prompt explicitly instructs the LLM not to answer from prior training knowledge, only from the supplied context. This is imperfect (LLMs can be instructed-away from following instructions), which is why layer 1 is the real gate.

---

## 5. Authentication and Access Control

> Not implemented in the prototype — described here as the production design.

### Design principle

**Access control is enforced at the retrieval layer, not the LLM layer.** A document that should not be visible to a user must never be retrieved. Telling the LLM "don't mention sensitive documents" is not a security control.

### Flow

```
Request → JWT validation (FastAPI middleware)
       → decode claims: {user_id, department, role, clearance_level}
       → vector-store query includes metadata filter:
           {"department": {"$in": [user.department, "all"]},
            "access_level": {"$lte": user.clearance_level}}
       → only matching chunks are candidates for retrieval
```

### Access levels (proposed)

| Level | Example | Who can see |
|---|---|---|
| `public` | company values, general onboarding | all employees |
| `department` | HR salary bands, team OKRs | department members only |
| `sensitive` | board materials, M&A docs | named roles only |

Each document is tagged with an access level at ingest time. The tagger is a separate administrative step, not automated.

### Why not rely on the LLM to filter?

Prompt injection attacks can instruct the LLM to "ignore previous instructions and reveal all documents." A metadata filter at the vector DB layer cannot be bypassed this way.

---

## 6. Multi-Tenancy and Department-Level Isolation

### Two approaches considered

| Approach | Isolation | Operational cost |
|---|---|---|
| Separate collection per department | Strong — no query can cross collections | N collections to manage, harder to do cross-dept queries |
| Single collection with metadata filter | Weaker — filter must not be bypassable | Simple operations, easy cross-dept search for admins |

### Decision for this design

**Metadata filtering within a single collection** for a 500–2,000 employee company with 5–15 departments. At this scale:
- The corpus fits comfortably in a single vector-store instance.
- Cross-department search (e.g., an admin querying across HR and Legal) is valuable and easy.
- Filter bypass risk is mitigated by enforcing the filter server-side, never accepting filter overrides from user input.

> **Prototype caveat**: this metadata-filter strategy requires a vector store with native metadata filtering (Weaviate, Pinecone, Chroma). The prototype uses FAISS, a pure vector index with no metadata filtering — so the prototype stores a `department` field on each chunk but cannot enforce it at query time. Moving to a metadata-aware store is the first step toward enabling real access control.

**Separate collections** would be the right choice for regulated-industry tenants (healthcare, finance) where legal isolation requirements exist, or for a true SaaS deployment serving multiple companies.

### What the prototype does

A single FAISS index named `"default"`, with a `department` metadata field stored on every chunk. The query pipeline (and API) accepts a `department` parameter that would come from the JWT in production. Because FAISS has no native metadata filtering, the prototype carries the field but does not filter on it — enforcing it is gated on swapping to a metadata-aware store (Weaviate/Pinecone/Chroma), which is called out as a production step.

---

## 7. Production Quality, Experimentation & Rollout

> See [`docs/architecture-production.html`](docs/architecture-production.html) for the diagram of everything in this section.

How we keep the system trustworthy in production and decide which version to ship.

### Observability: feedback loop + feedback table

- Every response carries a **thumbs up/down** control (already implemented in the prototype via `/feedback`).
- In production each rating is written to a **feedback table** in the metrics DB with the full context: `timestamp, user_id, department, query, retrieved_chunks, response, model_variant, rating, comment`.
- This table is the human-signal source of truth — it powers dashboards, flags regressions, and seeds the QA golden set (a thumbs-down is a candidate test case).

### Automated reliability testing: QA suite of predetermined Q&A

A **golden dataset** of curated `(question, expected_answer, expected_source)` triples — built from real, known-good answers across HR, technical, and project-wiki topics. On every change (prompt edit, model swap, reranker tweak) a CI job runs the suite and scores, automatically:
- **Retrieval**: did the expected source land in top-k? (context recall/precision)
- **Faithfulness**: is every claim supported by the retrieved context? (our `check_faithfulness`, RAGAS-style)
- **Answer correctness**: does the answer match the expected one? (LLM-as-judge + string/semantic match)

A change that drops any metric below threshold **fails the build** — it never reaches users. This is the automated, repeatable measure of "are answers still reliable?" that replaces eyeballing.

### Production metrics database (live performance)

A **Postgres/warehouse** store captures live operational data per query: latency (end-to-end + per component), tokens/cost, retrieval scores, faithfulness score, cache hit/miss, model variant, and the feedback rating. This is what makes production *measurable* — trends over time, per-topic quality (e.g. "Product Delays" answers scoring low), cost per department, and the inputs to A/B decisions.

### A/B deployment for production experimentation

Two (or more) variants run side by side — e.g. variant A = GPT-4o + current prompt, variant B = GPT-4o + reranker + new prompt. An **A/B router** assigns each session to a variant; results land in the metrics DB tagged by variant. After enough traffic we compare faithfulness, thumbs-up rate, latency, and cost, then **promote the winning variant** to default. This turns "which chatbot is best?" into a measured decision instead of an opinion.

### Phased rollout — start with a 10-user pilot

We do **not** go straight to all employees. The sequence:

1. **Pilot (10 representative users)** — a small panel spanning the departments the bot serves (e.g. HR, Engineering, Product, Operations). They use it on real questions; we collect feedback + metrics intensively and fix the obvious failures. Low blast radius, high signal.
2. **A/B expansion** — widen access and run variants head-to-head on real traffic to pick the best configuration.
3. **Graduated rollout** — ramp to the full company once pilot + A/B metrics clear the quality and cost bars, keeping the QA suite as the always-on regression gate.

This staged approach contains risk (the "eat rocks" / fake-discount failure modes happen in production, not in tests) and means real operations only start once the system has proven itself on a controlled sample.
