from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.memory.episodic.crud import store_memory, retrieve_memories
from src.memory.semantic.crud import store_fact, retrieve_facts

app = FastAPI(
    title="ThinkingMemory API",
    description="Agent-agnostic memory platform",
    version="0.1.0"
)

class MemoryStoreRequest(BaseModel):
    agent_id: str
    content: dict
    embedding: Optional[list[float]] = None
    extra_data: Optional[dict] = None

class FactStoreRequest(BaseModel):
    agent_id: str
    fact: str
    embedding: Optional[list[float]] = None
    confidence: float = 1.0
    source: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "ThinkingMemory API is running!"}

@app.post("/memory/store")
async def store_memory_endpoint(request: MemoryStoreRequest):
    try:
        return store_memory(
            agent_id=request.agent_id,
            content=request.content,
            embedding=request.embedding,
            extra_data=request.extra_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/retrieve/{agent_id}")
async def retrieve_memory_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_memories(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/store")
async def store_fact_endpoint(request: FactStoreRequest):
    try:
        return store_fact(
            agent_id=request.agent_id,
            fact=request.fact,
            embedding=request.embedding,
            confidence=request.confidence,
            source=request.source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/semantic/retrieve/{agent_id}")
async def retrieve_facts_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_facts(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))