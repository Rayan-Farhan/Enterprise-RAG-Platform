"""Admin operations router stub."""

from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/status", summary="Admin status")
async def admin_status() -> dict[str, str]:
    return {"message": "Admin status endpoint"}
