"""Documents management router stub."""

from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", summary="List documents")
async def list_documents() -> dict[str, str]:
    return {"message": "Document listing endpoint"}
