"""MinIO / S3 Object Storage implementation (ADR-003)."""

import io
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import AppSettings, get_settings
from app.core.exceptions import StorageException
from app.core.logging import get_logger
from app.storage.base import ObjectStorageProtocol, StoragePrefix

logger = get_logger("app.storage")


def build_storage_key(prefix: StoragePrefix | str, file_hash: str, filename_or_ext: str) -> str:
    """Build a content-addressed storage key conforming to ADR-003 layout.

    Example:
        build_storage_key(StoragePrefix.ORIGINAL, "a1b2c3...", "handbook.pdf")
        -> "original/a1b2c3....pdf"
    """
    prefix_val = prefix.value if isinstance(prefix, StoragePrefix) else str(prefix)
    ext = Path(filename_or_ext).suffix.lstrip(".")
    if not ext:
        ext = "bin"
    return f"{prefix_val}/{file_hash}.{ext}"


class MinIOStorageService(ObjectStorageProtocol):
    """S3-compatible object storage service backed by MinIO."""

    def __init__(self, settings: AppSettings | None = None, client: Minio | None = None) -> None:
        self.settings = settings or get_settings()
        self.bucket_name = self.settings.MINIO_BUCKET_NAME

        if client is not None:
            self.client = client
        else:
            self.client = Minio(
                endpoint=self.settings.MINIO_ENDPOINT,
                access_key=self.settings.MINIO_ACCESS_KEY,
                secret_key=self.settings.MINIO_SECRET_KEY,
                secure=self.settings.MINIO_SECURE,
            )

    def ensure_bucket_exists(self) -> None:
        """Bootstrap the default document storage bucket if not present."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info("MinIO bucket created successfully", bucket=self.bucket_name)
        except Exception as e:
            logger.warning(
                "Could not verify or create MinIO bucket",
                bucket=self.bucket_name,
                error=str(e),
            )

    def upload_file(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload byte payload or file stream to MinIO under the specified key."""
        try:
            stream: BinaryIO
            if isinstance(data, bytes):
                stream = io.BytesIO(data)
                length = len(data)
            else:
                stream = data
                # Seek end to find length if seekable, then reset
                try:
                    stream.seek(0, io.SEEK_END)
                    length = stream.tell()
                    stream.seek(0)
                except Exception:
                    length = -1

            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=stream,
                length=length,
                part_size=10 * 1024 * 1024 if length == -1 else 0,
                content_type=content_type,
                metadata=dict(metadata) if metadata else None,
            )
            logger.debug("Uploaded object to MinIO", key=key, bucket=self.bucket_name, size=length)
            return key
        except Exception as e:
            logger.error("Failed to upload object to MinIO", key=key, error=str(e))
            raise StorageException(f"Failed to upload object: {e}", key=key) from e

    def download_file(self, key: str) -> bytes:
        """Download object content as bytes."""
        try:
            response = self.client.get_object(self.bucket_name, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise StorageException(f"Object not found: {key}", key=key) from e
            raise StorageException(f"MinIO S3 error during download: {e}", key=key) from e
        except Exception as e:
            logger.error("Failed to download object from MinIO", key=key, error=str(e))
            raise StorageException(f"Failed to download object: {e}", key=key) from e

    def get_presigned_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        """Generate a presigned GET URL for secure object retrieval."""
        try:
            url: str = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=key,
                expires=timedelta(seconds=expires_in_seconds),
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned URL", key=key, error=str(e))
            raise StorageException(f"Failed to generate presigned URL: {e}", key=key) from e

    def delete_file(self, key: str) -> bool:
        """Remove an object from MinIO."""
        try:
            self.client.remove_object(self.bucket_name, key)
            return True
        except Exception as e:
            logger.error("Failed to delete object from MinIO", key=key, error=str(e))
            raise StorageException(f"Failed to delete object: {e}", key=key) from e

    def exists(self, key: str) -> bool:
        """Check if an object exists by statting it."""
        try:
            self.client.stat_object(self.bucket_name, key)
            return True
        except Exception:
            return False


_storage_service: MinIOStorageService | None = None


def get_storage_service() -> MinIOStorageService:
    """Return singleton MinIO storage service."""
    global _storage_service
    if _storage_service is None:
        _storage_service = MinIOStorageService()
    return _storage_service
