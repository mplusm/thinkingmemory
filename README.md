# ThinkingMemory

**The memory database agents recall from.** Point your agent at one endpoint; it
stores experience and gets back the *right* context — hybrid-retrieved, ranked,
and packed to a token budget — with recall that improves as the agent runs.

ThinkingMemory is agent-agnostic (works with any framework or LLM) and runs on
Postgres + pgvector with a local, offline embedding model by default.

## Why a "memory database" and not a vector store

A raw vector store gives you nearest-neighbours. An agent needs the *useful*
context for what it's doing right now, within a token budget. ThinkingMemory's
query primitive is **`recall`**:

- **Hybrid retrieval** — vector similarity + keyword (Postgres full-text) +
  recency, fused with Reciprocal Rank Fusion and weighted by salience.
- **Token-budget packing** — returns a ready-to-use, deduped context string with
  `[n]` citations, not a pile of rows.
- **Lifecycle** — each recall boosts the salience of what it surfaced, so useful
  memories rise over time.
- **Server-side embeddings** — you send text, not vectors.

One unified `Memory` substrate replaces the old four-layer split; the "layer"
(episodic / semantic / procedural / working) is now just a `mtype` tag on a row.

## Architecture

```
        any agent / framework / LLM
                  │   REST /v1   +   MCP tools
                  ▼
        ┌──────────────────────────┐
        │  Recall engine            │  embed → (vector | keyword | recency)
        │  hybrid → fuse → pack     │  → RRF + salience → token-budget packer
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  memory (Postgres)        │  one table: content+text+embedding,
        │  pgvector HNSW + FTS GIN  │  salience, bitemporal, provenance
        └──────────────────────────┘
```

- **Storage:** PostgreSQL 16+ / pgvector (`memory` table; HNSW + GIN indexes).
- **Embeddings:** `fastembed` + `BAAI/bge-small-en-v1.5` (384-dim, CPU, offline)
  by default; OpenAI optional. Pluggable via `EMBEDDING_PROVIDER`.
- **Working memory:** a separate Redis TTL scratchpad (`/working`).

## Getting Started

### Prerequisites
- Python 3.10+, PostgreSQL 16+ with `pgvector`, Redis 7+

### Install

```bash
git clone https://github.com/mplusm/thinkingmemory.git
cd thinkingmemory
python3 -m venv venv && source venv/bin/activate
pip install -e .            # installs fastembed + tiktoken
cp .env.example .env        # set DATABASE_URL etc.
python scripts/init_db.py   # creates the memory table + HNSW/GIN indexes
```

> The first `remember`/`recall` downloads the ~130 MB bge-small model once.
> Upgrading from the legacy four-layer schema? Run
> `python scripts/migrate_to_memory_db.py` to port existing rows (re-embedded)
> into the `memory` table.

### Run

```bash
uvicorn thinkingmemory.api.main:app --reload --port 8091
```

### Use it

```bash
# remember (text is embedded server-side)
curl -X POST localhost:8091/v1/remember -H 'Content-Type: application/json' -d '{
  "agent_id": "agent-1",
  "content": {"text": "To reset the analytics pipeline: stop workers, flush Redis, replay the Kafka offset."},
  "mtype": "procedural"
}'

# recall: intent in, packed + cited context out
curl -X POST localhost:8091/v1/recall -H 'Content-Type: application/json' -d '{
  "agent_id": "agent-1",
  "intent": "how do I restart the analytics pipeline?",
  "token_budget": 2000
}'
```

`recall` returns:

```json
{
  "context": "[1] To reset the analytics pipeline: stop workers, flush Redis, replay the Kafka offset.",
  "items": [{"citation": 1, "id": 42, "mtype": "procedural", "score": 0.83, "why": ["vector", "keyword"]}],
  "tokens_used": 22,
  "tokens_saved_vs_dump": 280
}
```

## API (`/v1`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/v1/remember` | Store a memory (embedded server-side) |
| `POST` | `/v1/remember/batch` | Store many in one batched embed call |
| `POST` | `/v1/recall` | **The primitive** — intent → packed, cited context |
| `POST` | `/v1/forget` | Forget a memory (soft by default; `hard` deletes) |
| `GET`  | `/v1/memory/{id}` | Fetch one memory |
| `GET`  | `/v1/trace/{id}` | Recursive provenance tree — why a memory is known |
| `GET`  | `/v1/timeline/{agent}?as_of=…` | What the agent believed at a point in time |
| `GET`  | `/v1/audit` | Append-only log of memory operations |
| `POST` | `/v1/maintenance/run` | Run the lifecycle cycle for an agent |
| `*`    | `/working/*` | Redis TTL scratchpad (short-term working memory) |

Interactive docs at `/docs`. All endpoints accept an optional `X-Tenant-ID`
header for per-tenant isolation (single-tenant when omitted).

## MCP server

Any MCP-capable agent can use memory as native tools (no HTTP glue):

```bash
pip install -e ".[mcp]"
thinkingmemory-mcp          # stdio
```

Tools: `remember`, `recall` (intent → packed context), `remember_fact`,
`remember_procedure`, `get_memory`, `forget`, and working-memory tools. Set a
default tenant with `THINKINGMEMORY_TENANT_ID`.

## Lifecycle (recall that improves over time)

Background maintenance is what makes this a memory *database*, not a vector
store. Run it on a schedule (`scripts/run_lifecycle.py --all`, or
`POST /v1/maintenance/run`):

- **decay** — salience fades by `e^(-decay_rate·Δt)` (per-`mtype` defaults:
  episodic fades fast, semantic/procedural slowly); recall counteracts it, so
  useful memories stay high and stale ones sink.
- **consolidate** ("sleep") — clusters of similar episodic memories are
  summarized into one semantic memory, linked to its sources via provenance.
- **forget** — low-salience, long-idle memories are soft-closed (recoverable),
  then hard-pruned after a grace period.
- **supersede** — near-duplicate semantic memories collapse to the newest
  (contradiction-lite), older ones closed and linked.

```bash
python scripts/run_lifecycle.py --agent agent-1 --interval-days 1
```

## Bitemporal & audit (enterprise)

Every memory records when it was *true* (`valid_from`/`valid_to`) and when we
*learned/closed* it (`created_at`/`superseded_at`), so you can ask what an agent
believed in the past and prove how it knows things:

- **`recall` with `as_of`** — retrieve against belief at a past moment.
- **`GET /v1/timeline/{agent}?as_of=…`** — a snapshot of everything believed then.
- **`GET /v1/trace/{id}`** — the recursive provenance tree (derived-from /
  superseded-by) behind a memory.
- **`GET /v1/audit`** — append-only log of remember/recall/forget/maintenance
  (toggle with `audit_enabled`).
- **Row-Level Security** — with `RLS_ENABLED=true` (run `scripts/enable_rls.py`
  once), Postgres itself restricts every query to the request's tenant via a
  per-session `app.tenant_id` GUC — defense-in-depth beneath app filtering. The
  policy allows access when unset, so single-tenant and admin/maintenance keep
  working.

## Evaluation harness

Retrieval quality is the product, so it's measured:

```bash
python eval/run_eval.py
```

On the bundled synthetic corpus, hybrid `recall` returns the gold memory **100%
of the time (recall@5)** versus **0% for a naive recency dump**, in **~68% fewer
tokens** than dumping the whole corpus. The harness is both the dev guardrail and
the ROI demo.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL URL | `postgresql://memory_user:...@localhost:5432/thinkingmemory` |
| `REDIS_URL` | Redis URL (working memory) | `redis://localhost:6379/0` |
| `EMBEDDING_PROVIDER` | `local` or `openai` | `local` |
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | Vector dimension (must match the model) | `384` |
| `OPENAI_API_KEY` | Required if `EMBEDDING_PROVIDER=openai` | — |
| `LOG_LEVEL` | Logging level | `INFO` |
| `AUDIT_ENABLED` | Append-only audit logging | `true` |
| `RLS_ENABLED` | Per-tenant Postgres Row-Level Security | `false` |

## Testing

```bash
pip install -e ".[dev]"
pytest          # runs against live Postgres + Redis; self-cleaning
```

## Roadmap

Done: unified substrate + hybrid recall (Phase 1); the lifecycle engine —
decay, consolidation, forgetting, supersession (Phase 2); and bitemporal
belief-over-time + provenance traces + audit log + per-tenant Row-Level Security
(Phase 3). Next: tenant partitioning, Apache AGE graph-hop recall, a scheduler
for lifecycle, cross-encoder reranking, and LLM-based fact extraction + NLI
contradiction detection. See `agent-db-plan.md`.

## License

MIT — see [LICENSE](LICENSE).
