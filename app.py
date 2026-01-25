import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_ai import ModelMessagesTypeAdapter
from src.debate_agent import invoke


import logfire

logfire.configure()  
logfire.instrument_pydantic_ai()

app = FastAPI(title="Debate Agent API", version="1.0.0")

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str

async def generate_stream(user_input: str):
    """Generate streaming response with events and outputs."""
    try:
        async for item in invoke(user_input):
            # DebateResponse 객체를 dict로 변환
            if item["type"] == "output" and hasattr(item["data"], "model_dump"):
                item["data"] = item["data"].model_dump()
            # JSON 라인 형식으로 전송
            yield json.dumps(item, ensure_ascii=False) + "\n"
    except Exception as e:
        error_item = {"type": "error", "message": str(e)}
        yield json.dumps(error_item, ensure_ascii=False) + "\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that streams debate agent responses.
    
    Request body:
    - message: str - User input message
    
    Response: Streaming JSON lines where each line is one of:
    - {"type": "event", "data": {...}} - Agent events (text, thinking, tool calls, etc.)
    - {"type": "output", "data": {...}} - Final debate response
    - {"type": "error", "message": "..."} - Error message
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return StreamingResponse(
        generate_stream(request.message),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Serve the main chat interface."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
