"""Object storage abstraction package (ADR-003)."""

from app.storage.base import ObjectStorageProtocol, StoragePrefix
from app.storage.minio_service import (
    MinIOStorageService,
    build_storage_key,
    get_storage_service,
)

__all__ = [
    "ObjectStorageProtocol",
    "StoragePrefix",
    "MinIOStorageService",
    "build_storage_key",
    "get_storage_service",
]
