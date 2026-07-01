# Cost Proposal — Corporate RAG Chatbot
Target company size: 500–2,000 employees

## Executive Summary

For **1,000 employees** (approximately 90,000 queries/month):

| | One-time build | Monthly run |
|---|---|---|
| **Recommended** | ~$33k (4-week MVP) | **~$278/mo** (tiered + cached, self-hosted vector store) |
| Unoptimized baseline | ~$134k (3-month build) | ~$704/mo (flat GPT-4o, no cache) |

- **Primary cost driver:** LLM tokens. **Primary lever:** model tiering (routing simple queries to GPT-4o-mini), yielding approximately 66% LLM savings.
- **Near-zero at this scale:** embeddings ($0, local) and the vector store ($0 self-hosted; managed Pinecone is a ~$50/mo plan floor rather than usage-based cost).
- **Per employee:** approximately $0.28–0.71/month, scaling roughly linearly with query volume.

> The figures above model the **current single-call RAG design**. The **3-month agentic target** (conversation agent, orchestrator, per-KB abilities) makes 3–5 LLM calls per query rather than one — approximately **2–2.3× the LLM cost (~$500–620/month all-in)**, largely absorbed by tiering the additional hops to small models. See [3-Month Agentic Architecture — Cost Delta](#3-month-agentic-architecture--cost-delta).

Detailed assumptions and calculations follow.

---

## Assumptions

All figures depend on the assumptions below; they are stated explicitly.

| Parameter | Value | Rationale |
|---|---|---|
| Employees | 1,000 | Mid-point of range |
| Daily active users (DAU) | 60% = 600 | Enterprise tool adoption is rarely 100% |
| Queries per active user per day | 5 | Based on Slack/search tool usage patterns |
| Queries per day | 3,000 | |
| Queries per month | ~90,000 | |
| Avg input tokens per query | 2,100 | System prompt (200) + context chunks (1,700) + user query (200) |
| Avg output tokens per query | 350 | Typical policy/FAQ answer length |
| Semantic cache hit rate | 15% | Repeated FAQs (e.g., "what is the PTO policy?") |
| Effective LLM calls per month | 76,500 | 90,000 × 0.85 |
| Corpus size | 10,000 documents | HR wiki, technical docs, project wikis |
| Avg chunks per document | 10 | At 512-char chunks |
| Total chunks indexed | 100,000 | |
| Embedding dimensions | 384 | all-MiniLM-L6-v2 |

---

## One-Time Build Costs

| Item | Hours | Rate | Cost |
|---|---|---|---|
| Senior AI engineer (RAG pipeline, evaluations, prompts) | 120 hrs | $150/hr | $18,000 |
| Backend engineer (FastAPI, auth, CI/CD) | 80 hrs | $130/hr | $10,400 |
| DevOps / infrastructure setup (cloud, secrets, monitoring) | 20 hrs | $130/hr | $2,600 |
| Document ingestion run (one-time operations) | 8 hrs | $80/hr | $640 |
| Security review | 10 hrs | $150/hr | $1,500 |
| **Total** | | | **~$33,000** |

> For a 3-month build — the **agentic, multi-knowledge-base redesign** (conversation and orchestrator agents, one RAG ability per KB, typed Pydantic I/O, two-tier memory) plus hybrid retrieval, role-based access, Slack integration, an evaluation harness, and the metrics/A/B infrastructure:
> 2 engineers × 12 weeks × 40 hrs × ~$140 blended = **~$134,000**. The additional agentic complexity is absorbed within this scope; it primarily shifts engineering effort toward orchestration and per-ability prompts rather than increasing headcount.

### Ingestion cost (one-time)

Embeddings run locally (sentence-transformers on CPU) — **$0 API cost**.

If OpenAI `text-embedding-3-small` were used instead ($0.02/1M tokens):
- 100,000 chunks × 128 tokens = 12.8M tokens → **$0.26** (negligible)

---

## Ongoing Monthly Costs

### LLM inference

> Rates verified Jun 2026: GPT-4o **$2.50 in / $10 out** per 1M tokens; GPT-4o-mini **$0.15 in / $0.60 out** per 1M; embeddings (text-embedding-3-small) **$0.02/1M** (the Batch API halves both LLM and embedding rates). Sources are listed at the end of this document.
>
> **Model landscape (a lever in itself):** GPT-4o is now effectively a prior-generation model; newer frontier models are *less expensive*: GPT-4.1 (~$2/1M in), GPT-5 (~$1.25/1M in). Anthropic Claude is a strong alternative (Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15, Opus 4.8 $5/$25 per 1M). **Because the organization operates on GCP, Gemini on Vertex AI is the consolidation option** — Gemini 2.x Flash is significantly less expensive (~$0.30–0.40/1M in), Pro is competitive with GPT-4o, and both keep data and billing within a single GCP project. GPT-4o is retained as a conservative, widely-recognized baseline for the calculations below; switching the synthesis model to GPT-4.1/GPT-5, Claude, or Gemini would *reduce* these figures, and all support 50%-off batch processing and prompt caching.

**Option A — GPT-4o only (flat)**

| | Calculation | Cost |
|---|---|---|
| Input | 76,500 calls × 2,100 tokens / 1M × $2.50 | $401 |
| Output | 76,500 calls × 350 tokens / 1M × $10.00 | $268 |
| **Subtotal** | | **$669/month** |

**Option B — Model tiering (recommended)**

Route approximately 70% of queries (simple lookups such as "what is the PTO policy?") to GPT-4o-mini and reserve 30% for complex multi-document reasoning.

| | Calls | Input cost | Output cost | Subtotal |
|---|---|---|---|---|
| GPT-4o-mini (70%) | 53,550 | 53,550 × 2,100 / 1M × $0.15 = $17 | 53,550 × 350 / 1M × $0.60 = $11 | $28 |
| GPT-4o (30%) | 22,950 | 22,950 × 2,100 / 1M × $2.50 = $120 | 22,950 × 350 / 1M × $10.00 = $80 | $200 |
| **Subtotal** | | | | **$228/month** |

**Model tiering reduces LLM cost by approximately 66%** — the single most impactful lever.

A simple query classifier (embedding cosine similarity against a set of "simple query" examples) routes queries before the LLM is invoked.

### Vector database

Verified against Pinecone's published pricing (Jun 2026): the **Standard plan has a $50/month minimum**, then $0.33/GB-month storage, $4 per 1M write units, and $16 per 1M read units (1 RU per GB scanned, minimum 0.25 RU/query). At this scale usage is negligible, but the plan floor still applies.

| Option | Storage | Query usage | Effective monthly |
|---|---|---|---|
| FAISS / Chroma / pgvector (self-hosted) | on the host | on the host | **~$0** (compute only) |
| Pinecone serverless (Standard) | 153 MB × $0.33/GB = $0.05 | 90k × 0.25 RU × $16/1M ≈ $0.36 | **~$50/month** (plan floor dominates) |

The usage-based cost of managed Pinecone here is **under $1/month**; the cost is the **$50 plan floor**. For a single mid-sized tenant, **self-hosted (Chroma / Weaviate / pgvector) is materially cheaper** (~$0 beyond the host); managed Pinecone becomes justified only when its scaling, operations, and high-availability features are required and the $50 floor is immaterial. (An earlier estimate of ~$5/month understated this; it is corrected here against published pricing.)

### Hosting

| Component | Spec | Cost |
|---|---|---|
| FastAPI server | AWS EC2 t3.medium (2 vCPU, 4 GB) | $30/month |
| or | GCP Cloud Run (scale-to-zero) | $10–20/month |
| Semantic cache (Redis) | ElastiCache t3.micro | $15/month |

Using Cloud Run: approximately $15/month.

### Observability

| Tool | Use | Cost |
|---|---|---|
| Langfuse (open-source tier) | LLM call tracing, latency, cost per query | $0 (self-hosted) or $30/month (cloud) |
| Cloud Monitoring / Datadog | Infrastructure metrics | ~$20/month |

### Monthly total summary

| Scenario | LLM | Hosting | Vector store (self-hosted) | Observability | **Total** |
|---|---|---|---|---|---|
| Flat GPT-4o, no caching | $669 | $15 | $0 | $20 | **~$704/month** |
| Tiered models + cache (recommended) | $228 | $30 | $0 | $20 | **~$278/month** |

> A self-hosted vector store (Chroma/Weaviate/pgvector) is approximately $0 beyond the host. Managed Pinecone adds its ~$50/month plan floor, bringing the recommended scenario to approximately $328.

**Per-employee-per-month (1,000 employees): approximately $0.28–0.70** (self-hosted vector store).

> The figures above are for the **current single-call design** (one LLM call per query). The 3-month agentic target costs more per query — see the next section.

---

## 3-Month Agentic Architecture — Cost Delta

The [agentic target](../ARCHITECTURE.md) replaces the single LLM call with a small pipeline of agents and abilities. The cost driver shifts from "one call per query" to **"N calls per query"**, making this the single largest change to the monthly cost.

### Calls per query (with model tiering)

| Hop | Model | Purpose | ~Tokens (in/out) |
|---|---|---|---|
| 1. Message → `Query` transform | GPT-4o-mini | inexpensive structuring | 500 / 100 |
| 2. Orchestrator routing | GPT-4o-mini | classify and select abilities | 600 / 50 |
| 3. Ability RAG (avg ~1.3/query) | tiered (mini/4o) | the retrieval-grounded answer | 2,100 / 350 |
| 4. Synthesis / select (multi-ability only) | GPT-4o | reason over candidates | 1,500 / 300 |
| 5. `Answer` → render | GPT-4o-mini | personalize wording | 800 / 350 |

The three small-model hops (1, 2, 5) are inexpensive; the significant cost remains the context-bearing ability call (3) and the occasional synthesis (4).

### Monthly estimate (76,500 effective queries after cache)

- **Small-model hops (1 + 2 + 5):** ~1,900 in + 500 out per query, all GPT-4o-mini
  → 76,500 × 1,900 / 1M × $0.15 + 76,500 × 500 / 1M × $0.60 ≈ **$22 + $23 = $45/mo**
- **Ability call (3), ~1.3/query, tiered 70/30:** ≈ 1.3 × the single-call tiered core ($228) ≈ **$300/mo**
- **Synthesis (4), ~30% of queries on GPT-4o:** 22,950 × (1,500 × $2.50 + 300 × $10) / 1M ≈ **$155/mo**
- **Agentic LLM subtotal ≈ $500/mo** (vs. $228 single-call) — approximately **2.2×**

### All-in monthly (agentic)

| Component | Single-call | Agentic (3-month) |
|---|---|---|
| LLM | $228 | ~$500 |
| Hosting (additional compute for orchestration) | $30 | $45 |
| Vector store (per-KB indexes, self-hosted) | $0 | $15 |
| Metrics DB (Postgres, live performance + feedback) | — | $25 |
| Observability | $20 | $30 |
| **Total** | **~$278** | **~$615/month** |

**Per-employee-per-month rises to approximately $0.62** — still well under one dollar.

### Levers specific to the agentic design

- **Aggressive tiering of the auxiliary hops** (transform/route/render must use small models) — this keeps the multiplier near 2.2× rather than 5×.
- **Skip synthesis for single-ability answers** — hop 4 is incurred only when the orchestrator fans out to multiple abilities.
- **Cache at the ability level**, not only the final answer — a repeated HR lookup can bypass hop 3.
- **The pilot phase adds negligible cost:** 10 users at approximately 50–100 queries/day amount to a few dollars/month in tokens, so the controlled rollout de-risks the expenditure at minimal cost.

---

## Cost Levers

1. **Model tiering** — the single largest lever, reducing LLM cost by approximately 66%. Route by query complexity rather than at random.
2. **Semantic caching** — store (query embedding → response) pairs in Redis. A cache hit costs approximately $0.0001 versus approximately $0.009 for a GPT-4o call; at a 15% hit rate, this saves approximately $100/month.
3. **Local embeddings** — sentence-transformers on CPU cost $0 per query. OpenAI embeddings at this volume would add approximately $2/month — minor, but local embeddings also remove an external runtime dependency.
4. **Context truncation** — reducing from 5 chunks to 3 chunks per query cuts input tokens by approximately 35%, saving approximately $140/month in the flat GPT-4o scenario.
5. **Quantization** — compress model parameters and vectors from 16/32-bit to 8-bit (or 1-bit) with minimal quality loss:
   - **Embedding quantization:** 8-bit vectors reduce vector-store storage/RAM by approximately 4× for a few points of recall@k. At larger corpora, this determines whether the index fits in RAM (fast) or spills to disk.
   - **LLM quantization:** relevant only for self-hosted models — an 8-bit model is smaller, faster, and less expensive to serve on the same GPU. Not applicable to API models such as GPT-4o.
   - **Matryoshka embeddings:** dimensions are ordered by information content, allowing retrieval with a short prefix (for example, the first 256 dimensions) for an inexpensive first pass, then re-ranking with full vectors. This reduces retrieval compute without re-indexing.
6. **Vector-store storage tiering** — keep the hot HNSW index in RAM, move rarely-accessed vectors to SSD, and store raw document text in inexpensive object storage. Combined with multi-tenancy (loading a department's vectors into RAM only when active), this keeps the most expensive memory tier small.
7. **Batch ingestion off-hours** — no direct per-token saving (embeddings are local), but this avoids API rate-limit throttling should the system later move to API-based embeddings.

---

## Scaling to 2,000 Employees

Doubling query volume: DAU 1,200 → 180,000 queries/month.

| Component | 1,000 employees | 2,000 employees |
|---|---|---|
| LLM (tiered) | $228 | $456 |
| Hosting (larger instance) | $30 | $60 |
| Vector store (self-hosted) | $0 | $0 |
| Observability | $20 | $25 |
| **Total** | **~$278** | **~$541** |

Costs scale roughly linearly with query volume. The vector-store cost barely moves, because the corpus size does not double simply because the user count does. The agentic design scales in the same manner — apply the approximately 2.2× LLM multiplier at each volume (approximately $615 → ~$1,150/month at 2,000 employees).

---

## Costs Not Included in This Model

- **Re-ranking inference:** a cross-encoder on CPU adds approximately $20–40/month for a small instance.
- **Document re-ingestion:** if the corpus updates weekly, re-embedding changed documents costs CPU time (local) — negligible.
- **Human evaluation:** a quarterly evaluation review (1 engineer, 8 hours) is approximately $1,200/year.
- **Compliance / legal review:** one-time, highly variable, not included.

---

## Pricing References (verified Jun 2026)

Rates are subject to change; the figures below were verified at the time of writing.

| Item | Rate used | Source |
|---|---|---|
| GPT-4o | $2.50 in / $10 out per 1M | [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) |
| GPT-4o-mini | $0.15 in / $0.60 out per 1M | [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) |
| text-embedding-3-small | $0.02 / 1M (Batch $0.01) | [OpenAI text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) |
| Newer/lower-cost LLMs | GPT-4.1 ~$2/1M in · GPT-5 ~$1.25/1M in | [OpenAI pricing overview](https://www.cloudzero.com/blog/openai-pricing/) |
| Claude (alternative) | Haiku 4.5 $1/$5 · Sonnet 4.6 $3/$15 · Opus 4.8 $5/$25 per 1M | [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Gemini on Vertex AI (GCP-native) | Flash ~$0.30–0.40/1M in · Pro competitive with GPT-4o | [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) |
| Pinecone Standard | $50/mo floor · $0.33/GB-mo · $4/1M WU · $16/1M RU | [Pinecone pricing](https://www.pinecone.io/pricing/) · [cost docs](https://docs.pinecone.io/guides/manage-cost/understanding-cost) |

Notes:
- **Batch API** (OpenAI and Anthropic) is approximately 50% less expensive for non-interactive work such as bulk ingestion or offline evaluation.
- **Prompt caching** further reduces repeated-prefix input cost (for example, the system prompt); Anthropic caches at approximately 90% off cached input.
- GPT-4o is retained as the calculation baseline for familiarity; moving synthesis to GPT-4.1/GPT-5 or Claude Sonnet/Haiku would lower the LLM figure.