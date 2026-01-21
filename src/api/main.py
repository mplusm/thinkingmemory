from fastapi import FastAPI
from src.memory.episodic.models import MemoryItem
from src.memory.episodic.crud import store_memory, retrieve_memories

app = FastAPI(
    title="ThinkingMemory API",
    description="Agent-agnostic memory platform",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "ThinkingMemory API is running!"}

@app.post("/memory/store")
async def store_memory_endpoint(item: MemoryItem):
    return store_memory(item)

@app.get("/memory/retrieve/{agent_id}")
async def retrieve_memory_endpoint(agent_id: str, limit: int = 10):
    return retrieve_memories(agent_id, limit)