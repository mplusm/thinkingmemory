# ThinkingMemory

An agent-agnostic memory platform for AI agents.

## Overview
ThinkingMemory provides a layered memory system (Working, Episodic, Semantic, Procedural) to enable AI agents to store, retrieve, and reason over past experiences.

## Features
- **Agent-agnostic**: Works with any AI agent framework (ReAct, AutoGPT, LangGraph, etc.).
- **Layered memory**: Separate layers for short-term, episodic, semantic, and procedural memory.
- **Reasoning-aware retrieval**: Agents specify intent, and the system retrieves relevant memory.
- **Forgetting strategies**: Automatically manage memory hygiene (time decay, relevance, etc.).

## Getting Started
### Prerequisites
- Python 3.11+
- PostgreSQL 16+ (with `pgvector` extension)
- Redis 7+

### Installation
```bash
git clone https://github.com/mplusm/thinkingmemory.git
cd thinkingmemory
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quickstart
1. Start the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```
2. Store a memory:
   ```bash
   curl -X POST http://localhost:8000/memory/store -H "Content-Type: application/json" -d '{
     "agent_id": "agent-123",
     "memory_type": "episodic",
     "content": {"goal": "Test memory", "outcome": "Success"}
   }'
   ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.