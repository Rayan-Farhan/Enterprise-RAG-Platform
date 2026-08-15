"""API v1 master router aggregating all sub-routers."""

from fastapi import APIRouter

from app.api.v1.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(admin_router)
api_v1_router.include_router(evaluation_router)
