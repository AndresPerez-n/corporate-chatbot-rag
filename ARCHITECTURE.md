# Architecture Overview — Corporate RAG Chatbot

> **This document covers two architectures**: the **current prototype** (what's built and runs today — Sections 1–6) and the **3-month target** (where this goes in production — Sections 7–8). Each prototype section flags what it does *not* do and why.
>
> **Diagrams are inline (Mermaid)** and render automatically on GitHub. Following the C4 convention, the system is shown at increasing zoom: the **prototype** flow below (current), then the **3-month target** at two levels — a container view and a zoom into the agentic core (Section 8).

## TL;DR

- **Query flow:** user → API (auth) → embed query → vector search → (re-rank) → inject top-k chunks into prompt → LLM → answer + citations.
- **Ingestion:** documents → parse → chunk (512 chars, 64 overlap) → embed (MiniLM) → store in vector index.
- **Retrieval:** dense vector search now; hybrid (BM25 + vector) + cross-encoder re-rank is the production design.
- **Access control:** enforced at the **retrieval layer** (metadata filter), never by asking the LLM to hide docs — that's not a security boundary.
- **Multi-tenancy:** single index + per-document `department`/`access_level` metadata filter; separate collections only for regulated isolation.
- **Prototype caveat:** FAISS (pure vector index) has no metadata filtering, so access control needs a metadata-aware store (Weaviate/Pinecone/Chroma) in production.
- **3-month target (Section 8):** evolves the single-index, single-agent prototype into an **agentic, multi-knowledge-base** system — a conversation agent + orchestrator routing across one specialized RAG ability per knowledge base, with personalization, two-tier memory, and typed (Pydantic/JSON) outputs.

Sections 1–6 expand the points above (the current prototype). Sections 7–8 cover the 3-month target.

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

> These production concerns wrap the agentic core shown in Section 8 (Diagram 8a places them in context).

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

---

## 8. Target Architecture (3-Month): Agentic, Multi-Knowledge-Base

This is the **3-month target**, not the prototype. The prototype (Sections 1–6) is deliberately a single-agent, single-index RAG; this section describes what it grows into. Two diagrams, zoomed in C4 style: first the whole system (container view), then the agentic core (component view).

### Diagram 8a — Container view (the whole system)

```mermaid
flowchart TB
    USR["Employee"]
    GW["API Gateway<br/>FastAPI · Okta JWT · rate limiting"]
    CORE["Agentic generation core<br/>conversation agent + orchestrator + abilities<br/>(detailed in 8b)"]
    KBS[("Vector stores<br/>one index per KB<br/>HR · Technical · Wiki")]
    DOCS[("Document store<br/>raw files, object storage")]
    META[("Metadata DB<br/>Postgres: catalog · access levels")]
    CACHE[("Redis<br/>semantic cache + sessions")]
    METRICS[("Metrics DB<br/>live perf + feedback")]
    ING["Ingestion pipeline<br/>connectors → parse → chunk → embed"]
    OBS["Observability · Eval · A/B<br/>Phoenix traces · RAGAS QA gate · dashboards"]

    USR -->|question| GW --> CORE
    CORE -->|answer + citations| GW --> USR
    CORE -->|retrieve chunks| KBS
    CORE -.->|fetch cited text| DOCS
    CORE -.->|access check| META
    CORE --> CACHE
    ING --> KBS
    ING --> DOCS
    ING --> META
    CORE -. live perf + traces .-> METRICS
    CORE -. traces .-> OBS
    METRICS --> OBS
```

### Diagram 8b — Agentic core (component view)

```mermaid
flowchart LR
    USR["Employee"]

    subgraph CONV["Conversation agent (front door)"]
        IN["Receive + load context"]
        TOQ["Transform to Query<br/>(Pydantic)"]
        REND["Render Answer<br/>personalized reply"]
    end

    subgraph CTX["Context"]
        PROF["User profile<br/>role · dept · purpose"]
        MEM["Memory<br/>last 5 verbatim + summary"]
    end

    subgraph ORCH["Orchestrator (planner)"]
        RT{{"Route + cache check<br/>+ ACL from JWT"}}
        SYN["Faithfulness gate +<br/>synthesize → Answer (Pydantic)"]
    end

    subgraph ABIL["Abilities — one RAG per knowledge base"]
        HR["HR ability"]
        TECH["Technical ability"]
        WIKI["Wiki ability"]
        TOOL["Tools (non-RAG)"]
    end

    USR --> IN --> TOQ -->|Query| RT
    RT --> HR
    RT --> TECH
    RT --> WIKI
    RT --> TOOL
    HR -->|result| SYN
    TECH -->|result| SYN
    WIKI -->|result| SYN
    TOOL -->|result| SYN
    SYN -->|Answer| REND --> USR
    PROF -. personalize + access .-> IN
    MEM -. recent + summary .-> TOQ
```

> Typed hand-offs at every hop — `Query` → `AbilityResult` → `Answer` (Pydantic/JSON) — enforce formatting and make hallucinations checkable (reject/retry on malformed or unsupported output). Routing/transform use small models, synthesis uses a larger one (tiering), keeping the multi-hop cost to ~2.2× the single-call design.

### Why agentic, and why now

The brief explicitly lists **several** distinct knowledge bases — HR policies, technical documentation, project wikis. The prototype flattens them into one index, which is fine for a proof-of-concept but suboptimal: an HR policy question and a database-migration question want different retrieval settings, different prompts, and different access rules.

The target instead models **each knowledge base as its own specialized RAG "ability"** behind an orchestrator. This lets us tune chunking/retrieval/prompt per domain, scope access per ability, and add non-RAG abilities (e.g. "who's on-call?" via an API) without bolting them onto a monolith.

### Two-agent design

**1. Conversation Agent (the front door).** Owns the dialogue with the employee. Responsibilities:
- Receive the raw user message and hold the conversational context (user profile + history).
- **Transform the message into a structured query** — a typed `Query` object (Pydantic), not free text.
- Take the orchestrator's structured answer and **render it into a natural, personalized reply** for the user.

**2. Orchestrator Agent (the planner/router).** Never talks to the user directly. Responsibilities:
- Receive the structured `Query`.
- **Route** to the right ability or abilities (e.g. HR ability, technical ability), possibly several in parallel.
- Collect the candidate answers, run the **faithfulness/grounding gate**, and **select or synthesize** the final structured answer.
- Return a typed `Answer` object to the conversation agent.

### Abilities / skills (one RAG per knowledge base)

Each ability is a self-contained tool with a typed interface, its own retriever config, its own prompt, and its own access scope:

| Ability | Knowledge base / source | Type |
|---|---|---|
| HR ability | HR policy KB | RAG |
| Technical ability | Engineering docs KB | RAG |
| Project-wiki ability | Project/wiki KB | RAG |
| Directory / on-call ability | PagerDuty / HR system | non-RAG tool |
| Ticketing ability | Jira / ITSM | non-RAG action |

Adding a new domain = adding a new ability, not reworking a monolith.

### Per-ability data pipeline: storage, chunking, retrieval

Section 2 (chunking) and Section 3 (retrieval) describe the *prototype's single* pipeline. In the agentic target each ability owns its own, tuned to its content. This is the part that was implicit before — made concrete here.

**Where the data lives (three stores, not one).** "The database" is really three layers, because raw files, searchable vectors, and metadata have different needs:

| Layer | Holds | Tech | Why separate |
|---|---|---|---|
| Document store | the raw source files (PDF, MD, HTML) + versions | object storage (S3 / GCS) | cheap, durable; cited text is fetched from here at answer time |
| Vector store (one index per KB) | chunk embeddings + per-chunk metadata (`source`, `dept`, `access_level`) | Weaviate / Pinecone / pgvector | metadata-aware filtering enables access control; per-KB tuning |
| Metadata / relational DB | doc catalog, access levels, ingestion state, versions | Postgres | drives incremental re-ingest and access decisions |

**Chunking — tuned per domain (not one size):**

| Ability | Chunking strategy | Why |
|---|---|---|
| HR policies | section/heading-aware, ~512 tokens, keep a clause whole | policies are short, self-contained rules; don't split a rule |
| Technical docs | structure-aware, larger (~800–1,000 tokens), keep code blocks + tables intact (Unstructured.io) | code/tables lose meaning when split mid-block |
| Project wikis | heading-based, medium, carry the page-title as a prefix on every chunk | wiki pages are long and multi-topic; the title disambiguates |

All use overlap to avoid boundary loss (Section 2), and re-chunk incrementally when the source doc changes (hash-based change detection).

**Retrieval — hybrid + re-rank, per-ability config:**
- **Hybrid** (BM25 + dense vector, fused with RRF) — the production strategy from Section 3, now applied inside each ability so it can weight keyword vs. semantic differently (technical docs lean keyword for exact API/error names; HR leans semantic).
- **Cross-encoder re-rank** on the merged candidates before they go to the LLM.
- **Metadata filter first** — the ACL from `UserContext` is applied at query time so out-of-scope documents are never even retrieved (Section 5).
- **Per-ability `top_k` and threshold** — an ability with a small, precise KB (HR) can run a tighter threshold than a broad one (wiki).

The orchestrator doesn't know these details — each ability exposes the same typed interface (`Query in → AbilityResult out`), so retrieval internals stay encapsulated.

### End-to-end flow

```
User message
  → Conversation Agent  (load user profile + history)
      → builds Query{intent, text, filters, user_ctx}      (Pydantic)
  → Orchestrator Agent
      → routes to ability(ies)
          → HR ability   → RAG over HR KB        → AbilityResult
          → Tech ability → RAG over Technical KB → AbilityResult
      → faithfulness gate + select / synthesize
      → Answer{text, citations, confidence, faithfulness}   (Pydantic)
  → Conversation Agent  (render, personalize)
  → user
```

This is the **conditional/router workflow** pattern (orchestrator routes; each ability is a tool; an evaluator/citation step finishes), with the conversation agent acting as a dedicated input/output transformer.

### Context system (personalization)

A typed **`UserContext`** travels with every request: `{user_id, role, department, clearance_level, purpose}`. It does two jobs:
- **Personalization** — the same question yields a role-appropriate answer (an engineer asking about "deployment" vs. someone in HR).
- **Access control** — role/clearance decides which abilities and which documents are even reachable (the retrieval-layer enforcement from Section 5, now per-ability).

### History management (two-tier)

To keep continuity without unbounded token growth:
- **Short-term (verbatim):** the **last 5 messages** stay live in the agent's working context.
- **Long-term (summarized):** everything older is compressed into a rolling **summary history** — a running synopsis the agent carries instead of the full transcript.

This is the prototype's rolling-window idea (Section 4) generalized: recent turns precise, older turns summarized.

### Typed outputs (Pydantic / JSON) for formatting + anti-hallucination

Every hop passes a **validated, typed object** — `Query`, `AbilityResult`, `Answer` — not loose text:
- **Formatting guarantee:** the final answer always has the same shape (text, citations, confidence, faithfulness), which the UI and downstream systems can rely on.
- **Hallucination guard:** structured outputs are *checkable* — if the model returns malformed JSON, missing citations, or claims with no supporting chunk, validation **rejects and retries**. Combined with the faithfulness self-check (Section 7 / already prototyped), this is the layered "no hallucination + consistent format" guarantee.

### What this adds over the prototype (and the cost of it)

| Dimension | Prototype (now) | Agentic target (3-month) |
|---|---|---|
| Agents | Single RAG call | Conversation agent + orchestrator + N abilities |
| Knowledge bases | One flat FAISS index | One specialized RAG per KB |
| Storage | One index; raw text inline | 3 layers: object store (files) + per-KB vector store + Postgres metadata |
| Chunking | Uniform 512/64 | Per-domain (section-aware HR, code/table-aware technical, heading-based wiki) |
| Retrieval | Dense vector, top-k + threshold | Hybrid (BM25+vector+RRF) + cross-encoder re-rank + metadata ACL, per-ability config |
| Personalization | None | `UserContext` (role/purpose) drives answer + access |
| Memory | Last-3-turn window | Last-5 verbatim + summarized long-term |
| I/O | Free-text | Typed Pydantic/JSON at every hop |

The cost is real — multiple LLM hops per query mean more latency and tokens (mitigated by model tiering: small models for routing/transform, larger for synthesis). That overhead only pays off once there are genuinely distinct knowledge bases and personalization needs, which is exactly why it's the **3-month** target and not the prototype.
