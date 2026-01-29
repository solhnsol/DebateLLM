import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from src.session_manager import SessionManager

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

app = FastAPI(title="Debate Agent API", version="1.0.0")

# 전역 세션 매니저
session_manager = SessionManager()


class CreateSessionRequest(BaseModel):
    user_id: str
    topic: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


@app.post("/sessions")
async def create_session(request: CreateSessionRequest):
    session = session_manager.create_session(
        user_id=request.user_id,
        topic=request.topic
    )
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "topic": session.topic,
        "user_role": session.user_role,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": 0
    }


@app.get("/sessions")
async def list_sessions(user_id: Optional[str] = None):
    sessions = session_manager.list_sessions()
    if user_id:
        sessions = [s for s in sessions if session_manager.get_session(s["session_id"]).user_id == user_id]
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "topic": session.topic,
        "user_role": session.user_role,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(session.chat_history)
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    history = session_manager.get_chat_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": [msg.model_dump() for msg in history]
    }


async def generate_stream(session_id: str, user_input: str):
    try:
        async for item in session_manager.send_message(session_id, user_input):
            if item.get("type") == "history":
                continue
            yield json.dumps(item, ensure_ascii=False) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"


@app.post("/sessions/{session_id}/chat")
async def chat_in_session(session_id: str, request: ChatRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return StreamingResponse(
        generate_stream(session_id, request.message),
        media_type="application/x-ndjson"
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
