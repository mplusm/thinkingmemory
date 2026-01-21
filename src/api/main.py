from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.memory.episodic.crud import store_memory, retrieve_memories, forget_old_memories, forget_low_relevance_memories
from src.memory.semantic.crud import store_fact, retrieve_facts, forget_low_confidence_facts
from src.memory.procedural.crud import store_procedure, retrieve_procedures, update_procedure_success_rate, forget_low_success_procedures

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

class ProcedureStoreRequest(BaseModel):
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict]
    success_rate: float = 1.0
    version: int = 1

class ProcedureUpdateRequest(BaseModel):
    success_rate: float

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

@app.post("/procedural/store")
async def store_procedure_endpoint(request: ProcedureStoreRequest):
    try:
        return store_procedure(
            agent_id=request.agent_id,
            name=request.name,
            description=request.description,
            steps=request.steps,
            success_rate=request.success_rate,
            version=request.version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/procedural/retrieve/{agent_id}")
async def retrieve_procedures_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_procedures(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/procedural/update/{procedure_id}")
async def update_procedure_endpoint(procedure_id: int, request: ProcedureUpdateRequest):
    try:
        return update_procedure_success_rate(procedure_id, request.success_rate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/forget/old/{agent_id}")
async def forget_old_memories_endpoint(agent_id: str, days: int = 30):
    try:
        deleted_count = forget_old_memories(agent_id, days)
        return {"message": f"Deleted {deleted_count} old memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/forget/low-relevance/{agent_id}")
async def forget_low_relevance_memories_endpoint(agent_id: str, relevance_threshold: float = 0.5):
    try:
        deleted_count = forget_low_relevance_memories(agent_id, relevance_threshold)
        return {"message": f"Deleted {deleted_count} low-relevance memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/semantic/forget/low-confidence/{agent_id}")
async def forget_low_confidence_facts_endpoint(agent_id: str, confidence_threshold: float = 0.5):
    try:
        deleted_count = forget_low_confidence_facts(agent_id, confidence_threshold)
        return {"message": f"Deleted {deleted_count} low-confidence facts for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/procedural/forget/low-success/{agent_id}")
async def forget_low_success_procedures_endpoint(agent_id: str, success_threshold: float = 0.5):
    try:
        deleted_count = forget_low_success_procedures(agent_id, success_threshold)
        return {"message": f"Deleted {deleted_count} low-success procedures for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))