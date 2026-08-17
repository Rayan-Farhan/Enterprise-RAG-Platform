"""FastAPI application factory and lifecycle entrypoint (ADR-001, ADR-030)."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import AppException, ProblemDetail
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware


async def _warm_dependencies(logger: Any) -> None:
    """Construct the blocking service clients before serving traffic.

    Building the Qdrant and MinIO clients costs ~2-3s on first use (module import
    plus a version-compatibility request). Paying that inside the first readiness
    probe makes the probe exceed its timeout and report a healthy dependency as
    unreachable. Warming here is best-effort: a dependency that is down must not
    prevent startup, because readiness — not startup — is what gates traffic.
    """
    import asyncio

    def warm() -> None:
        from app.retrieval.vector_store import get_vector_store
        from app.storage.minio_service import get_storage_service

        get_vector_store().client.get_collections()
        get_storage_service().ensure_bucket_exists()

    try:
        await asyncio.wait_for(asyncio.to_thread(warm), timeout=20.0)
        logger.info("Dependency clients warmed")
    except Exception as exc:  # noqa: BLE001 - startup must survive a cold dependency
        logger.warning(
            "Dependency warm-up incomplete; readiness will report the details",
            error=str(exc),
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(debug=settings.DEBUG, app_env=settings.APP_ENV)
    logger = get_logger("app.main")

    logger.info(
        "Application starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
        inference_profile=settings.INFERENCE_PROFILE,
    )

    await _warm_dependencies(logger)

    yield

    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Middlewares
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers (RFC-7807 Problem Details)
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger = get_logger("app.exceptions")
        logger.warning(
            "Application exception handled",
            error_code=exc.code,
            status_code=exc.status_code,
            message=exc.message,
            details=exc.details,
        )
        problem = ProblemDetail(
            title=exc.code.replace("_", " ").title(),
            status=exc.status_code,
            detail=exc.message,
            instance=str(request.url.path),
            code=exc.code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=problem.model_dump(exclude_none=True),
            headers={"Content-Type": "application/problem+json"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger = get_logger("app.exceptions")
        logger.info("Request validation failed", errors=exc.errors())

        errors_summary = [
            {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        problem = ProblemDetail(
            title="Request Validation Error",
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The request body or parameters failed validation.",
            instance=str(request.url.path),
            code="VALIDATION_ERROR",
            invalid_params=errors_summary,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=problem.model_dump(exclude_none=True),
            headers={"Content-Type": "application/problem+json"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = get_logger("app.exceptions")
        logger.exception("Unhandled internal exception occurred", exc_info=exc)

        problem = ProblemDetail(
            title="Internal Server Error",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please contact support with the trace ID.",
            instance=str(request.url.path),
            code="INTERNAL_SERVER_ERROR",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=problem.model_dump(exclude_none=True),
            headers={"Content-Type": "application/problem+json"},
        )

    # Root endpoint (Task 0.1 acceptance criteria)
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """Root endpoint returning service name, version, and status."""
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "environment": settings.APP_ENV,
            "inference_profile": settings.INFERENCE_PROFILE,
        }

    # Mount API v1
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
