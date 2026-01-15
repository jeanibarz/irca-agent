import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

DATA_DIR = Path("data/sessions")


class Message(BaseModel):
    role: str
    content: str


class ConversationMetadata(BaseModel):
    id: str
    title: str
    updated_at: str


class Conversation(ConversationMetadata):
    messages: list[Message]


class CreateConversationRequest(BaseModel):
    title: str | None = None


class UpdateConversationRequest(BaseModel):
    messages: list[Message]
    title: str | None = None


@router.get("/conversations", response_model=list[ConversationMetadata])
async def list_conversations() -> list[ConversationMetadata]:
    """List all saved conversations ordered by recency."""
    conversations = []
    if not DATA_DIR.exists():
        return []

    for file_path in DATA_DIR.glob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)
                # Fallback for old files if any
                if "updated_at" not in data:
                    data["updated_at"] = datetime.now().isoformat()

                conversations.append(
                    ConversationMetadata(
                        id=data["id"], title=data.get("title", "Untitled"), updated_at=data["updated_at"]
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to read conversation {file_path}: {e}")

    # Sort by updated_at desc
    conversations.sort(key=lambda x: x.updated_at, reverse=True)
    return conversations


@router.post("/conversations", response_model=Conversation)
async def create_conversation(req: CreateConversationRequest) -> Conversation:
    """Create a new conversation."""
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    conversation = Conversation(id=conv_id, title=req.title or "New Chat", updated_at=now, messages=[])

    file_path = DATA_DIR / f"{conv_id}.json"
    with open(file_path, "w") as f:
        json.dump(conversation.dict(), f, indent=2)

    return conversation


@router.get("/conversations/{conv_id}", response_model=Conversation)
async def get_conversation(conv_id: str) -> Conversation:
    """Get a specific conversation."""
    file_path = DATA_DIR / f"{conv_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")

    with open(file_path) as f:
        data = json.load(f)
        return Conversation(**data)


@router.post("/conversations/{conv_id}", response_model=Conversation)
async def update_conversation(conv_id: str, req: UpdateConversationRequest) -> Conversation:
    """Update conversation messages and optionally title."""
    file_path = DATA_DIR / f"{conv_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")

    with open(file_path) as f:
        data = json.load(f)

    # Update fields
    if req.title:
        data["title"] = req.title

    # Auto-title if it's "New Chat" and we have messages now
    if data["title"] == "New Chat" and len(req.messages) > 0:
        first_msg = req.messages[0].content
        data["title"] = (first_msg[:30] + "...") if len(first_msg) > 30 else first_msg

    data["messages"] = [m.dict() for m in req.messages]
    data["updated_at"] = datetime.now().isoformat()

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return Conversation(**data)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str) -> dict[str, str]:
    """Delete a conversation."""
    file_path = DATA_DIR / f"{conv_id}.json"
    if file_path.exists():
        file_path.unlink()
        return {"status": "success", "message": "Deleted"}
    raise HTTPException(status_code=404, detail="Conversation not found")
