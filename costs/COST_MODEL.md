# Cost Model — Corporate RAG Chatbot
Target company size: 500–2,000 employees

## TL;DR

For **1,000 employees** (~90k queries/month):

| | One-time build | Monthly run |
|---|---|---|
| **Headline** | ~$33k (4-week MVP) | **~$278/mo** (tiered + cached, self-hosted vector DB) |
| Naive alternative | ~$134k (3-month build) | ~$704/mo (flat GPT-4o, no cache) |

- **Biggest cost driver:** LLM tokens. **Biggest lever:** model tiering (route simple queries to GPT-4o-mini) → ~66% LLM savings.
- **Near-free at this scale:** embeddings ($0, local) and vector DB ($0 self-hosted; managed Pinecone is a ~$50/mo plan floor, not usage).
- **Per employee:** ~$0.28–0.71/month. Scales roughly linearly with query volume.

> The numbers above model the **current single-call RAG design**. The **3-month agentic target** (conversation agent + orchestrator + per-KB abilities) makes 3–5 LLM calls per query instead of one — roughly **2–2.3× the LLM cost (~$500–620/month all-in)**, mostly absorbed by tiering the extra hops to small models. See [3-Month Agentic Architecture — Cost Delta](#3-month-agentic-architecture--cost-delta).

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

> At a 3-month build — the **agentic, multi-knowledge-base redesign** (conversation + orchestrator agents, one RAG ability per KB, typed Pydantic I/O, two-tier memory) plus hybrid retrieval, role-based access, Slack integration, eval harness, and the metrics/A/B infrastructure:
> 2 engineers × 12 weeks × 40 hrs × ~$140 blended = **~$134,000**. The added agentic complexity is absorbed within this scope; it mostly shifts engineering time toward orchestration and per-ability prompts rather than raising the headcount.

### Ingestion cost (one-time)

Embeddings run locally (sentence-transformers on CPU) — **$0 API cost**.

If using OpenAI `text-embedding-3-small` instead ($0.02/1M tokens):
- 100,000 chunks × 128 tokens = 12.8M tokens → **$0.26** (negligible)

---

## Ongoing Monthly Costs

### LLM inference

> Rates verified Jun 2026: GPT-4o **$2.50 in / $10 out** per 1M tokens; GPT-4o-mini **$0.15 in / $0.60 out** per 1M; embeddings (text-embedding-3-small) **$0.02/1M** (Batch API halves both LLM and embedding rates). Sources at the bottom of this doc.
>
> **Model landscape (a lever in itself):** GPT-4o is now effectively legacy — newer frontier models are *cheaper*: GPT-4.1 (~$2/1M in), GPT-5 (~$1.25/1M in). Anthropic Claude is a strong alternative (Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15, Opus 4.8 $5/$25 per 1M). We keep GPT-4o in the math below as a conservative, widely-known baseline; switching the synthesis model to GPT-4.1/GPT-5 or Claude Sonnet/Haiku would *reduce* these numbers, and all support 50%-off batch + prompt caching.

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

Verified against Pinecone's published pricing (Jun 2026): the **Standard plan has a $50/month minimum**, then $0.33/GB-month storage, $4 per 1M write units, $16 per 1M read units (1 RU per GB scanned, min 0.25 RU/query). At our scale the *usage* is trivial — but you pay the floor.

| Option | Storage | Query usage | Effective monthly |
|---|---|---|---|
| FAISS / Chroma / pgvector (self-hosted) | on the host | on the host | **~$0** (compute only) |
| Pinecone serverless (Standard) | 153 MB × $0.33/GB = $0.05 | 90k × 0.25 RU × $16/1M ≈ $0.36 | **~$50/month** (plan floor dominates) |

The usage-based cost of managed Pinecone here is **under $1/month** — the bill is the **$50 plan floor**. So for a single mid-sized tenant, **self-hosted (Chroma / Weaviate / pgvector) is materially cheaper** (~$0 beyond the host); managed Pinecone only pays off once you want its scaling/ops/HA and the $50 floor is noise. (An earlier draft under-counted Pinecone at ~$5/month — corrected here against the real pricing.)

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

| Scenario | LLM | Hosting | Vector DB (self-hosted) | Observability | **Total** |
|---|---|---|---|---|---|
| Flat GPT-4o, no caching | $669 | $15 | $0 | $20 | **~$704/month** |
| Tiered models + cache (recommended) | $228 | $30 | $0 | $20 | **~$278/month** |

> Self-hosted vector DB (Chroma/Weaviate/pgvector) is ~$0 beyond the host. Choosing **managed Pinecone** instead adds its ~$50/month plan floor → ~$328 for the recommended scenario.

**Per-employee-per-month (1,000 employees): ~$0.28 – $0.70** (self-hosted vector DB).

> The figures above are for the **current single-call design** (one LLM call per query). The 3-month agentic target costs more per query — see the next section.

---

## 3-Month Agentic Architecture — Cost Delta

The [agentic target](../ARCHITECTURE.md) replaces the single LLM call with a small pipeline of agents/abilities. The cost driver shifts from "one call per query" to **"N calls per query"**, so this is the single biggest change to the monthly bill.

### Calls per query (with model tiering)

| Hop | Model | Why | ~Tokens (in/out) |
|---|---|---|---|
| 1. Message → `Query` transform | GPT-4o-mini | cheap structuring | 500 / 100 |
| 2. Orchestrator routing | GPT-4o-mini | classify + pick abilities | 600 / 50 |
| 3. Ability RAG (avg ~1.3/query) | tiered (mini/4o) | the actual retrieval-grounded answer | 2,100 / 350 |
| 4. Synthesis / select (multi-ability only) | GPT-4o | reason over candidates | 1,500 / 300 |
| 5. `Answer` → render | GPT-4o-mini | personalize wording | 800 / 350 |

The three small-model hops (1, 2, 5) are cheap; the expensive part is still the context-bearing ability call (3) and the occasional synthesis (4).

### Monthly estimate (76,500 effective queries after cache)

- **Small-model hops (1 + 2 + 5):** ~1,900 in + 500 out per query, all GPT-4o-mini
  → 76,500 × 1,900 / 1M × $0.15 + 76,500 × 500 / 1M × $0.60 ≈ **$22 + $23 = $45/mo**
- **Ability call (3), ~1.3/query, tiered 70/30:** ≈ 1.3 × the single-call tiered core ($228) ≈ **$300/mo**
- **Synthesis (4), ~30% of queries on GPT-4o:** 22,950 × (1,500 × $2.50 + 300 × $10) / 1M ≈ **$155/mo**
- **Agentic LLM subtotal ≈ $500/mo** (vs. $228 single-call) — about **2.2×**

### All-in monthly (agentic)

| Component | Single-call | Agentic (3-month) |
|---|---|---|
| LLM | $228 | ~$500 |
| Hosting (more compute for orchestration) | $30 | $45 |
| Vector DB (per-KB indexes, self-hosted) | $0 | $15 |
| Metrics DB (Postgres, live perf + feedback) | — | $25 |
| Observability | $20 | $30 |
| **Total** | **~$278** | **~$615/month** |

**Per-employee-per-month rises to ~$0.62** — still well under a dollar.

### Levers specific to the agentic design

- **Aggressive tiering of the plumbing hops** (transform/route/render must be small models) — this is what keeps it at ~2.2× and not ~5×.
- **Skip synthesis for single-ability answers** — only pay for hop 4 when the orchestrator actually fans out.
- **Cache at the ability level**, not just the final answer — a repeated HR lookup can short-circuit hop 3.
- **Pilot phase is essentially free**: 10 users ≈ 50–100 queries/day → a few dollars/month in tokens, so the controlled rollout adds negligible cost while de-risking the spend.

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
| Vector DB (self-hosted) | $0 | $0 |
| Observability | $20 | $25 |
| **Total** | **~$278** | **~$541** |

Costs scale roughly linearly with query volume. The vector DB cost barely moves because the corpus size doesn't double just because the user count does. The agentic design scales the same way — apply the ~2.2× LLM multiplier at each volume (so ~$615 → ~$1,150/month at 2,000 employees).

---

## What We're Not Modelling

- **Re-ranking inference**: a cross-encoder on CPU adds ~$20–40/month for a small instance.
- **Document re-ingestion**: if the corpus updates weekly, re-embedding changed documents costs CPU time (local) — negligible.
- **Human eval cost**: a quarterly eval review (1 engineer, 8 hours) = ~$1,200/year.
- **Compliance / legal review**: one-time, highly variable, not included.

---

## Pricing References (verified Jun 2026)

Rates can change; these were checked when this doc was written.

| Item | Rate used | Source |
|---|---|---|
| GPT-4o | $2.50 in / $10 out per 1M | [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) |
| GPT-4o-mini | $0.15 in / $0.60 out per 1M | [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) |
| text-embedding-3-small | $0.02 / 1M (Batch $0.01) | [OpenAI text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) |
| Newer/cheaper LLMs | GPT-4.1 ~$2/1M in · GPT-5 ~$1.25/1M in | [OpenAI pricing overview](https://www.cloudzero.com/blog/openai-pricing/) |
| Claude (alternative) | Haiku 4.5 $1/$5 · Sonnet 4.6 $3/$15 · Opus 4.8 $5/$25 per 1M | [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Pinecone Standard | $50/mo floor · $0.33/GB-mo · $4/1M WU · $16/1M RU | [Pinecone pricing](https://www.pinecone.io/pricing/) · [cost docs](https://docs.pinecone.io/guides/manage-cost/understanding-cost) |

Notes:
- **Batch API** (OpenAI and Anthropic) is ~50% off for non-interactive work like bulk ingestion or offline eval.
- **Prompt caching** further reduces repeated-prefix input cost (e.g. the system prompt) — Anthropic caches at ~90% off cached input.
- GPT-4o is kept as the math baseline for familiarity; moving synthesis to GPT-4.1/GPT-5 or Claude Sonnet/Haiku would lower the LLM line.
