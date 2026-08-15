"""Evaluation runs and metrics router stub."""

from fastapi import APIRouter

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/runs", summary="List evaluation runs")
async def list_evaluation_runs() -> dict[str, str]:
    return {"message": "Evaluation runs listing endpoint"}
