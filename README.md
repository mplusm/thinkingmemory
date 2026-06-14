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
| `GET`  | `/v1/trace/{id}` | Provenance — why a memory is known |
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

## Testing

```bash
pip install -e ".[dev]"
pytest          # runs against live Postgres + Redis; self-cleaning
```

## Roadmap

Lifecycle workers (decay, consolidation, contradiction resolution),
cross-encoder reranking, graph-hop recall, bitemporal `/timeline` & provenance
traces, and per-tenant RLS + partitioning. See `agent-db-plan.md`.

## License

MIT — see [LICENSE](LICENSE).
