"""Chat and streaming completion router stub."""

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", summary="Send query and receive answer")
async def chat_completion() -> dict[str, str]:
    return {"message": "Chat completion endpoint"}
