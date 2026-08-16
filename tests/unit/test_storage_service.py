"""Unit tests for MinIO object storage abstraction (Task 2.3, ADR-003)."""

from unittest.mock import MagicMock

import pytest

from app.core.config import AppSettings
from app.storage.base import ObjectStorageProtocol, StoragePrefix
from app.storage.minio_service import MinIOStorageService, build_storage_key


def test_build_storage_key() -> None:
    """Verify storage key path conforms to ADR-003 prefix layout."""
    key = build_storage_key(StoragePrefix.ORIGINAL, "a1b2c3d4e5", "employee_handbook.pdf")
    assert key == "original/a1b2c3d4e5.pdf"

    table_key = build_storage_key(StoragePrefix.TABLES, "tbl_99", "table_data.json")
    assert table_key == "tables/tbl_99.json"


def test_storage_service_protocol_conformance() -> None:
    """Verify MinIOStorageService implements ObjectStorageProtocol."""
    mock_client = MagicMock()
    service = MinIOStorageService(client=mock_client)
    assert isinstance(service, ObjectStorageProtocol)


def test_upload_file_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify uploading bytes invokes client.put_object with correct arguments."""
    mock_client = MagicMock()
    service = MinIOStorageService(
        settings=AppSettings(MINIO_BUCKET_NAME="test-bucket"), client=mock_client
    )

    data = b"Sample PDF document binary content"
    key = "original/sample_doc.pdf"

    returned_key = service.upload_file(key, data, content_type="application/pdf")
    assert returned_key == key
    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args.kwargs
    assert call_kwargs["bucket_name"] == "test-bucket"
    assert call_kwargs["object_name"] == key
    assert call_kwargs["length"] == len(data)
    assert call_kwargs["content_type"] == "application/pdf"


def test_download_file() -> None:
    """Verify download_file retrieves and reads object stream."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.read.return_value = b"Downloaded document content"
    mock_client.get_object.return_value = mock_response

    service = MinIOStorageService(
        settings=AppSettings(MINIO_BUCKET_NAME="test-bucket"), client=mock_client
    )

    result = service.download_file("original/test.pdf")
    assert result == b"Downloaded document content"
    mock_response.close.assert_called_once()
    mock_response.release_conn.assert_called_once()


def test_get_presigned_url() -> None:
    """Verify generating presigned URL."""
    mock_client = MagicMock()
    mock_client.presigned_get_object.return_value = "http://minio:9000/test-bucket/original/test.pdf?sig=123"

    service = MinIOStorageService(
        settings=AppSettings(MINIO_BUCKET_NAME="test-bucket"), client=mock_client
    )

    url = service.get_presigned_url("original/test.pdf", expires_in_seconds=1800)
    assert "sig=123" in url
    mock_client.presigned_get_object.assert_called_once()


def test_ensure_bucket_exists() -> None:
    """Verify bucket bootstrap logic."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False

    service = MinIOStorageService(
        settings=AppSettings(MINIO_BUCKET_NAME="test-bucket"), client=mock_client
    )
    service.ensure_bucket_exists()

    mock_client.bucket_exists.assert_called_once_with("test-bucket")
    mock_client.make_bucket.assert_called_once_with("test-bucket")
