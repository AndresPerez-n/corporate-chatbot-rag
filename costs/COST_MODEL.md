# Cost Model — Corporate RAG Chatbot
Target company size: 500–2,000 employees

## TL;DR

For **1,000 employees** (~90k queries/month):

| | One-time build | Monthly run |
|---|---|---|
| **Headline** | ~$33k (4-week MVP) | **~$283/mo** (tiered + cached) |
| Naive alternative | ~$134k (3-month build) | ~$709/mo (flat GPT-4o, no cache) |

- **Biggest cost driver:** LLM tokens. **Biggest lever:** model tiering (route simple queries to GPT-4o-mini) → ~66% LLM savings.
- **Near-free at this scale:** embeddings ($0, local) and vector DB ($0 self-hosted FAISS, ~$5/mo managed).
- **Per employee:** ~$0.28–0.71/month. Scales roughly linearly with query volume.

Full assumptions and math below.

---

## Assumptions

State these clearly — the numbers are only as good as the assumptions underneath them.

| Parameter | Value | Rationale |
|---|---|---|
| Employees | 1,000 | Mid-point of range |
| Daily active users (DAU) | 60% = 600 | Enterprise tool adoption is rarely 100% |
| Queries per active user per day | 5 | Based on Slack/search tool usage patterns |
| Queries per day | 3,000 | |
| Queries per month | ~90,000 | |
| Avg input tokens per query | 2,100 | System prompt (200) + context chunks (1,700) + user query (200) |
| Avg output tokens per query | 350 | Typical policy/FAQ answer length |
| Semantic cache hit rate | 15% | Repeated FAQs (e.g., "what's the PTO policy?") |
| Effective LLM calls per month | 76,500 | 90,000 × 0.85 |
| Corpus size | 10,000 documents | HR wiki, tech docs, project wikis |
| Avg chunks per document | 10 | At 512-char chunks |
| Total chunks indexed | 100,000 | |
| Embedding dimensions | 384 | all-MiniLM-L6-v2 |

---

## One-Time Build Costs

| Item | Hours | Rate | Cost |
|---|---|---|---|
| Senior AI engineer (RAG pipeline, evals, prompts) | 120 hrs | $150/hr | $18,000 |
| Backend engineer (FastAPI, auth, CI/CD) | 80 hrs | $130/hr | $10,400 |
| DevOps / infra setup (AWS, secrets, monitoring) | 20 hrs | $130/hr | $2,600 |
| Document ingestion run (one-time ops) | 8 hrs | $80/hr | $640 |
| Security review | 10 hrs | $150/hr | $1,500 |
| **Total** | | | **~$33,000** |

> At a 3-month build (higher scope: streaming, role-based access, Slack integration, eval harness):
> 2 engineers × 12 weeks × 40 hrs × ~$140 blended = **~$134,000**

### Ingestion cost (one-time)

Embeddings run locally (sentence-transformers on CPU) — **$0 API cost**.

If using OpenAI `text-embedding-3-small` instead ($0.02/1M tokens):
- 100,000 chunks × 128 tokens = 12.8M tokens → **$0.26** (negligible)

---

## Ongoing Monthly Costs

### LLM inference

**Option A — GPT-4o only (flat)**

| | Calculation | Cost |
|---|---|---|
| Input | 76,500 calls × 2,100 tokens / 1M × $2.50 | $401 |
| Output | 76,500 calls × 350 tokens / 1M × $10.00 | $268 |
| **Subtotal** | | **$669/month** |

**Option B — Model tiering (recommended)**

Route ~70% of queries (simple lookups: "what's the PTO?") to GPT-4o-mini and keep 30% for complex multi-document reasoning.

| | Calls | Input cost | Output cost | Subtotal |
|---|---|---|---|---|
| GPT-4o-mini (70%) | 53,550 | 53,550 × 2,100 / 1M × $0.15 = $17 | 53,550 × 350 / 1M × $0.60 = $11 | $28 |
| GPT-4o (30%) | 22,950 | 22,950 × 2,100 / 1M × $2.50 = $120 | 22,950 × 350 / 1M × $10.00 = $80 | $200 |
| **Subtotal** | | | | **$228/month** |

**Model tiering saves ~66% on LLM costs** — the most impactful single lever.

A simple query classifier (embedding cosine similarity to a set of "simple query" examples) routes queries before the LLM is called.

### Vector database

| Option | Storage | Queries | Total |
|---|---|---|---|
| FAISS / Chroma (self-hosted, included in server) | $0 | $0 | $0 |
| Pinecone serverless | 100,000 × 384d × 4B = 153MB → ~$0.01 | 90,000 × 5 reads / 1M × $0.40 = $0.18 | ~$5/month |

Prototype uses FAISS (self-hosted, $0). Production would move to a metadata-aware store — self-hosted Chroma/Weaviate stays at ~$0 infra beyond the host, or Pinecone adds managed scalability, metadata filtering, and backups for ~$5/month.

### Hosting

| Component | Spec | Cost |
|---|---|---|
| FastAPI server | AWS EC2 t3.medium (2 vCPU, 4 GB) | $30/month |
| OR | GCP Cloud Run (scale-to-zero) | $10–20/month |
| Semantic cache (Redis) | ElastiCache t3.micro | $15/month |

Using Cloud Run: ~$15/month.

### Observability

| Tool | Use | Cost |
|---|---|---|
| Langfuse (open-source tier) | LLM call tracing, latency, cost per query | $0 (self-hosted) or $30/month (cloud) |
| Datadog / CloudWatch | Infra metrics | ~$20/month |

### Monthly total summary

| Scenario | LLM | Hosting | Vector DB | Observability | **Total** |
|---|---|---|---|---|---|
| Flat GPT-4o, no caching | $669 | $15 | $5 | $20 | **$709/month** |
| Tiered models + cache (recommended) | $228 | $30 | $5 | $20 | **$283/month** |

**Per-employee-per-month (1,000 employees): $0.28 – $0.71**

---

## Cost Levers

1. **Model tiering** — single biggest lever. 66% LLM cost reduction. Route by query complexity, not randomly.
2. **Semantic caching** — store (query embedding → response) pairs in Redis. A cache hit costs ~$0.0001 vs. ~$0.009 for a GPT-4o call. At 15% hit rate: saves ~$100/month.
3. **Local embeddings** — sentence-transformers on CPU costs $0 per query. OpenAI embeddings at this volume would add ~$2/month — minor, but it also removes an external runtime dependency.
4. **Context truncation** — reducing from 5 chunks to 3 chunks per query cuts input tokens by ~35%, saving ~$140/month on the flat GPT-4o scenario.
5. **Quantization** — compress model parameters and vectors from 16/32-bit down to 8-bit (or even 1-bit) with minimal quality loss:
   - **Embedding quantization**: 8-bit vectors cut vector-DB storage/RAM ~4x for a few points of recall@k. At larger corpora this is the difference between fitting the index in RAM (fast) or spilling to disk.
   - **LLM quantization**: only relevant if self-hosting — an 8-bit model is smaller, faster, and cheaper to serve on the same GPU. Not applicable to API models like GPT-4o.
   - **Matryoshka embeddings**: dimensions ordered by information, so you can retrieve with a short prefix (e.g. first 256 dims) for a cheap first pass, then re-rank with full vectors. Cuts retrieval compute without re-indexing.
6. **Vector-DB storage tiering** — keep the hot HNSW index in RAM, move rarely-accessed vectors to SSD, and store raw document text in cheap object storage. Combined with multi-tenancy (load a department's vectors into RAM only when active), this keeps the most expensive memory tier small.
7. **Batch ingestion off-hours** — no direct per-token saving (embeddings are local), but avoids API rate-limit throttling if switching to API-based embeddings later.

---

## Scaling to 2,000 Employees

Double the query volume: DAU 1,200 → 180,000 queries/month.

| Component | 1,000 employees | 2,000 employees |
|---|---|---|
| LLM (tiered) | $228 | $456 |
| Hosting (larger instance) | $30 | $60 |
| Vector DB | $5 | $5 |
| Observability | $20 | $25 |
| **Total** | **$283** | **$546** |

Costs scale roughly linearly with query volume. The vector DB cost barely moves because the corpus size doesn't double just because the user count does.

---

## What We're Not Modelling

- **Re-ranking inference**: a cross-encoder on CPU adds ~$20–40/month for a small instance.
- **Document re-ingestion**: if the corpus updates weekly, re-embedding changed documents costs CPU time (local) — negligible.
- **Human eval cost**: a quarterly eval review (1 engineer, 8 hours) = ~$1,200/year.
- **Compliance / legal review**: one-time, highly variable, not included.
