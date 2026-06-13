# ThinkingMemory

An agent-agnostic memory platform for AI agents.

## Overview
ThinkingMemory provides a layered memory system (Working, Episodic, Semantic, Procedural) to enable AI agents to store, retrieve, and reason over past experiences.

## Features
- **Agent-agnostic**: Works with any AI agent framework (ReAct, AutoGPT, LangGraph, etc.).
- **Layered memory**: Separate layers for short-term, episodic, semantic, and procedural memory.
- **Reasoning-aware retrieval**: Agents specify intent, and the system retrieves relevant memory.
- **Forgetting strategies**: Automatically manage memory hygiene (time decay, relevance, etc.).
- **MCP-native**: Ships an [MCP](https://modelcontextprotocol.io) server so any MCP-capable agent can `remember`/`recall` with zero glue (see [MCP Server](#mcp-server)).
- **Multi-tenant**: Every API call can be scoped to a tenant via the `X-Tenant-ID` header; storage is isolated per tenant.
- **Indexed similarity search**: Embedding columns are fixed-dimension with HNSW indexes for fast vector recall.

## Memory Layers

### Working Memory (Redis)
Short-term key-value store with TTL-based auto-expiry.

### Episodic Memory
Stores past actions, decisions, and outcomes with vector embeddings for similarity search.

### Semantic Memory
Structured knowledge storage with specialized models:
- **Facts** — generic text facts with confidence scores
- **Data Sources** — databases, warehouses, and external systems
- **Data Tables** — tables/views within data sources, with schema and row count info
- **Data Columns** — columns with types, foreign keys, sample values, and lineage tracking
- **Knowledge Entities** — generic structured knowledge (API schemas, configs, system components, concepts)

### Procedural Memory
Behavioral knowledge storage with specialized models:
- **Procedures** — named step-by-step workflows with success rates
- **User Preferences** — how the user likes things done (upsert semantics by agent+category+key)
- **Workflow Habits** — observed behavioral patterns with separate success/failure counts

## Getting Started
### Prerequisites
- Python 3.11+
- PostgreSQL 16+ (with `pgvector` extension)
- Redis 7+

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mplusm/thinkingmemory.git
   cd thinkingmemory
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` to include your database credentials:
     ```plaintext
     DATABASE_URL=postgresql://memory_user:your_password@localhost:5432/thinkingmemory
     ```

> **Embeddings:** Embedding vectors are stored at a fixed dimension (`EMBEDDING_DIM`, default `1536`). Set this to match your embedding model *before* initializing the database. Embeddings are supplied by the caller — ThinkingMemory stores and indexes them but does not generate them. If you are upgrading an existing database created before dimensions were pinned, run `python scripts/migrate_vector_dims.py` once to size the columns and build the HNSW indexes.

### Quickstart
1. Initialize the database (creates tables, the `pgvector` extension, and HNSW indexes):
   ```bash
   python scripts/init_db.py
   ```

2. Start the API:
   ```bash
   uvicorn thinkingmemory.api.main:app --reload --port 8091
   ```

3. Store a memory:
   ```bash
   curl -X POST http://localhost:8091/memory/store -H "Content-Type: application/json" -d '{
     "agent_id": "agent-123",
     "content": {"goal": "Test memory", "outcome": "Success"}
   }'
   ```

4. Retrieve memories:
   ```bash
   curl -X GET http://localhost:8091/memory/retrieve/agent-123
   ```

5. Store a fact:
   ```bash
   curl -X POST http://localhost:8091/semantic/store -H "Content-Type: application/json" -d '{
     "agent_id": "agent-123",
     "fact": "The sky is blue",
     "confidence": 0.95
   }'
   ```

6. Retrieve facts:
   ```bash
   curl -X GET http://localhost:8091/semantic/retrieve/agent-123
   ```

7. Store a procedure:
   ```bash
   curl -X POST http://localhost:8091/procedural/store -H "Content-Type: application/json" -d '{
     "agent_id": "agent-123",
     "name": "Test Procedure",
     "steps": [{"step": 1, "action": "Do something"}]
   }'
   ```

8. Retrieve procedures:
   ```bash
   curl -X GET http://localhost:8091/procedural/retrieve/agent-123
   ```

9. Update procedure success rate:
   ```bash
   curl -X PATCH http://localhost:8091/procedural/update/1 -H "Content-Type: application/json" -d '{
     "success_rate": 0.95
   }'
   ```

10. Forget old memories:
    ```bash
    curl -X DELETE "http://localhost:8091/memory/forget/old/agent-123?days=30"
    ```

11. Forget low-relevance memories:
    ```bash
    curl -X DELETE "http://localhost:8091/memory/forget/low-relevance/agent-123?relevance_threshold=0.5"
    ```

12. Forget low-confidence facts:
    ```bash
    curl -X DELETE "http://localhost:8091/semantic/forget/low-confidence/agent-123?confidence_threshold=0.5"
    ```

13. Forget low-success procedures:
    ```bash
    curl -X DELETE "http://localhost:8091/procedural/forget/low-success/agent-123?success_threshold=0.5"
    ```

### Semantic Memory — Data Models

14. Store a data source:
    ```bash
    curl -X POST http://localhost:8091/semantic/sources/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "source_name": "production_postgres",
      "source_type": "postgresql",
      "description": "Main production database"
    }'
    ```

15. Store a data table:
    ```bash
    curl -X POST http://localhost:8091/semantic/tables/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "table_name": "users",
      "data_source_id": 1,
      "schema_name": "public",
      "table_type": "table",
      "description": "Core user accounts",
      "tags": ["core", "pii"]
    }'
    ```

16. Store a data column with lineage:
    ```bash
    curl -X POST http://localhost:8091/semantic/columns/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "column_name": "user_id",
      "data_table_id": 1,
      "data_type": "integer",
      "is_primary_key": true,
      "lineage": {
        "upstream": [{"source": "raw.events", "column": "user_id", "transform": "direct"}],
        "downstream": [{"target": "analytics.user_stats", "column": "uid"}]
      }
    }'
    ```

17. Retrieve a table with all its columns:
    ```bash
    curl -X GET http://localhost:8091/semantic/tables/with-columns/1
    ```

18. Retrieve column lineage:
    ```bash
    curl -X GET http://localhost:8091/semantic/columns/lineage/1
    ```

19. Store a knowledge entity:
    ```bash
    curl -X POST http://localhost:8091/semantic/knowledge/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "entity_type": "api_endpoint",
      "name": "/api/v1/users",
      "description": "Returns paginated user list",
      "properties": {"method": "GET", "auth": "bearer"},
      "relationships": [{"type": "depends_on", "target_name": "auth_service"}],
      "confidence": 1.0
    }'
    ```

20. Retrieve knowledge by type:
    ```bash
    curl -X GET "http://localhost:8091/semantic/knowledge/retrieve/agent-123?entity_type=api_endpoint"
    ```

### Procedural Memory — Preferences & Habits

21. Store a user preference (upserts on same agent+category+key):
    ```bash
    curl -X POST http://localhost:8091/procedural/preferences/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "category": "code_style",
      "key": "indent_style",
      "value": "spaces_4",
      "confidence": 1.0,
      "source": "explicit"
    }'
    ```

22. Retrieve a specific preference:
    ```bash
    curl -X GET "http://localhost:8091/procedural/preferences/by-key/agent-123?category=code_style&key=indent_style"
    ```

23. Store a workflow habit:
    ```bash
    curl -X POST http://localhost:8091/procedural/habits/store -H "Content-Type: application/json" -d '{
      "agent_id": "agent-123",
      "habit_name": "morning_standup_query",
      "pattern": {"trigger": "9am daily", "steps": ["query jira", "summarize blockers"], "context": "standup"},
      "tags": ["daily", "standup"]
    }'
    ```

24. Increment a habit (track success/failure):
    ```bash
    curl -X POST http://localhost:8091/procedural/habits/increment/1 -H "Content-Type: application/json" -d '{
      "success": true
    }'
    ```

25. Forget unused habits:
    ```bash
    curl -X DELETE "http://localhost:8091/procedural/habits/forget/unused/agent-123?max_frequency=1"
    ```

## Multi-tenancy

All endpoints accept an optional `X-Tenant-ID` header. When present, every store,
retrieve, and forget operation is scoped to that tenant, and data from other
tenants is never returned. When the header is omitted, the API runs in
single-tenant mode (no tenant filtering). For example:

```bash
curl -X POST http://localhost:8091/memory/store \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme-corp" \
  -d '{"agent_id": "agent-123", "content": {"goal": "Test"}}'
```

Wrapper applications can override the `get_tenant_id` dependency (e.g. to derive
the tenant from an authenticated API key) without modifying the routers.

## MCP Server

ThinkingMemory ships a [Model Context Protocol](https://modelcontextprotocol.io)
server so MCP-capable agents can use memory as native tools — no HTTP glue. It
talks directly to the database/Redis and is configured from the same `.env`.

Install the MCP extra and run it over stdio:

```bash
pip install -e ".[mcp]"      # or: pip install mcp
thinkingmemory-mcp           # or: python -m thinkingmemory.mcp_server
```

Register it with an MCP client (e.g. Claude Desktop) by adding to its config:

```json
{
  "mcpServers": {
    "thinkingmemory": {
      "command": "thinkingmemory-mcp",
      "env": {
        "DATABASE_URL": "postgresql://memory_user:your_password@localhost:5432/thinkingmemory",
        "REDIS_URL": "redis://localhost:6379/0",
        "EMBEDDING_DIM": "1536"
      }
    }
  }
}
```

### Available tools

| Tool | Layer | Purpose |
|------|-------|---------|
| `remember` | Episodic | Store an experience/event |
| `recall` | Episodic | Recall most recent memories (no embedding needed) |
| `recall_similar` | Episodic | Vector-similarity recall (requires a query embedding) |
| `forget_old_memories` | Episodic | Delete memories older than N days |
| `memory_stats` | Episodic | Stats about an agent's memory |
| `store_fact` / `recall_facts` | Semantic | Store/recall durable facts |
| `store_procedure` / `recall_procedures` | Procedural | Store/recall reusable workflows |
| `set_working_memory` / `get_working_memory` / `list_working_memory` | Working | TTL-backed scratchpad |

Each tool takes an optional `tenant_id` argument; a process-wide default can be
set with `THINKINGMEMORY_TENANT_ID`.

## API Reference

All endpoints are documented in the interactive Swagger UI at `/docs` when the API is running.

### Endpoint Overview

| Prefix | Description |
|--------|-------------|
| `/memory/` | Episodic memory (store, retrieve, forget, similar) |
| `/working/` | Working memory (key-value with TTL) |
| `/semantic/store`, `/semantic/retrieve/`, `/semantic/similar/` | Semantic facts |
| `/semantic/sources/` | Data source registry |
| `/semantic/tables/` | Data table metadata |
| `/semantic/columns/` | Data column definitions and lineage |
| `/semantic/knowledge/` | Generic knowledge entities |
| `/procedural/store`, `/procedural/retrieve/`, `/procedural/similar/` | Procedures |
| `/procedural/preferences/` | User preferences (upsert semantics) |
| `/procedural/habits/` | Workflow habits with success/failure tracking |

## Environment Variables
The following environment variables can be set in a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL database URL | `postgresql://memory_user:your_password@localhost:5432/thinkingmemory` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `EMBEDDING_DIM` | Embedding vector dimension (must match your model) | `1536` |
| `LOG_LEVEL` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) | `INFO` |
| `DEBUG` | Enable SQL query logging | `false` |
| `THINKINGMEMORY_TENANT_ID` | Default tenant for the MCP server | _(unset)_ |

## Testing

The test suite runs against a live Postgres + Redis (it cleans up after itself
and skips gracefully if the services are unreachable):

```bash
pip install -e ".[dev]"
pytest
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.