"""Object storage protocol and storage key prefix layout (ADR-003)."""

from enum import StrEnum
from typing import BinaryIO, Protocol, runtime_checkable


class StoragePrefix(StrEnum):
    """Locked S3/MinIO bucket prefix layout (ADR-003)."""

    ORIGINAL = "original"
    NORMALIZED = "normalized"
    PAGES = "pages"
    IMAGES = "images"
    TABLES = "tables"
    OCR = "ocr"
    DERIVED = "derived"
    EXPORTS = "exports"
    EVALUATION = "evaluation"


@runtime_checkable
class ObjectStorageProtocol(Protocol):
    """S3-compatible object storage service protocol."""

    def upload_file(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file or byte stream to the object store and return its key."""
        ...

    def download_file(self, key: str) -> bytes:
        """Retrieve file content by key."""
        ...

    def get_presigned_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        """Generate a time-limited presigned URL for downloading or viewing an object."""
        ...

    def delete_file(self, key: str) -> bool:
        """Delete an object by key."""
        ...

    def exists(self, key: str) -> bool:
        """Check whether an object key exists in the bucket."""
        ...

    def ensure_bucket_exists(self) -> None:
        """Bootstrap bucket if it does not exist on startup."""
        ...
