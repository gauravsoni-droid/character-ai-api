import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError

load_dotenv()

TOKEN = os.getenv("TOKEN")
DEFAULT_CHARACTER_ID = "dxDIxJrhmT4IE22Gih1OqvER0JZpDkTD5bvui3fh-Qs"

if TOKEN is None:
    raise RuntimeError("TOKEN environment variable is not set")

# ---------------------------------------------------------------------------
# In‑memory store for active chat sessions
# ---------------------------------------------------------------------------
# Each entry: { session_id: { "client", "chat", "character_id", "username" } }
sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Lifespan – clean‑up all sessions on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Close every open session on shutdown
    for sid, data in sessions.items():
        try:
            await data["client"].close_session()
        except Exception:
            pass
    sessions.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Character.AI Chat API",
    description="REST API wrapper around PyCharacterAI for managing chat sessions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CreateChatRequest(BaseModel):
    character_id: Optional[str] = Field(
        default=None,
        description="Character.AI character ID. Uses the default character when omitted.",
    )


class CreateChatResponse(BaseModel):
    session_id: str
    chat_id: str
    character_id: str
    username: str
    greeting: str
    greeting_author: str


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The message to send to the character.")


class MessageResponse(BaseModel):
    author: str
    text: str


class SessionInfo(BaseModel):
    session_id: str
    chat_id: str
    character_id: str
    username: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
async def root():
    """Health check / welcome endpoint."""
    return {
        "status": "ok",
        "message": "Character.AI Chat API is running 🚀",
        "docs": "/docs",
    }


@app.get("/sessions", response_model=list[SessionInfo], tags=["Sessions"])
async def list_sessions():
    """List all active chat sessions."""
    return [
        SessionInfo(
            session_id=sid,
            chat_id=data["chat"].chat_id,
            character_id=data["character_id"],
            username=data["username"],
        )
        for sid, data in sessions.items()
    ]


@app.post("/sessions", response_model=CreateChatResponse, status_code=201, tags=["Sessions"])
async def create_session(body: CreateChatRequest = CreateChatRequest()):
    """
    Create a new chat session with a Character.AI character.

    Returns the session ID, chat ID, and the character's greeting message.
    """
    character_id = body.character_id or DEFAULT_CHARACTER_ID

    try:
        client = await get_client(token=TOKEN)
        me = await client.account.fetch_me()

        chat, greeting_message = await client.chat.create_chat(character_id)

        session_id = uuid.uuid4().hex[:12]
        sessions[session_id] = {
            "client": client,
            "chat": chat,
            "character_id": character_id,
            "username": me.name,
        }

        greeting_text = greeting_message.get_primary_candidate().text
        greeting_author = greeting_message.author_name

        return CreateChatResponse(
            session_id=session_id,
            chat_id=chat.chat_id,
            character_id=character_id,
            username=me.name,
            greeting=greeting_text,
            greeting_author=greeting_author,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {exc}")


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse, tags=["Chat"])
async def send_message(session_id: str, body: SendMessageRequest):
    """
    Send a message in an existing chat session and receive the full response
    (non‑streaming).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    data = sessions[session_id]
    client = data["client"]
    chat = data["chat"]
    character_id = data["character_id"]

    try:
        answer = await client.chat.send_message(
            character_id,
            chat.chat_id,
            body.message,
            streaming=True,
        )

        # Consume the stream and return the final message
        final_message = None
        async for msg in answer:
            final_message = msg

        if final_message is None:
            raise HTTPException(status_code=502, detail="No response received from character.")

        return MessageResponse(
            author=final_message.author_name,
            text=final_message.get_primary_candidate().text,
        )

    except SessionClosedError:
        sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail="Session has been closed.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {exc}")


@app.post("/sessions/{session_id}/messages/stream", tags=["Chat"])
async def send_message_stream(session_id: str, body: SendMessageRequest):
    """
    Send a message and stream the response back as **Server‑Sent Events (SSE)**.

    Each event contains a JSON payload `{ "author": "...", "text": "..." }` where
    `text` is the *full* accumulated text so far.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    data = sessions[session_id]
    client = data["client"]
    chat = data["chat"]
    character_id = data["character_id"]

    async def event_generator():
        import json

        try:
            answer = await client.chat.send_message(
                character_id,
                chat.chat_id,
                body.message,
                streaming=True,
            )

            async for msg in answer:
                text = msg.get_primary_candidate().text
                payload = json.dumps({"author": msg.author_name, "text": text})
                yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

        except SessionClosedError:
            sessions.pop(session_id, None)
            yield f'data: {json.dumps({"error": "Session closed"})}\n\n'
        except Exception as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def close_session(session_id: str):
    """Close an active chat session and release resources."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    data = sessions.pop(session_id)
    try:
        await data["client"].close_session()
    except Exception:
        pass

    return {"status": "closed", "session_id": session_id}