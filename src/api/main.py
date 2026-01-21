from fastapi import FastAPI

app = FastAPI(
    title="ThinkingMemory API",
    description="Agent-agnostic memory platform",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "ThinkingMemory API is running!"}