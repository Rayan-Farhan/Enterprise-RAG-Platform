"""Search router stub."""

from fastapi import APIRouter

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", summary="Execute search")
async def execute_search() -> dict[str, str]:
    return {"message": "Search endpoint"}
