import aioboto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _session() -> aioboto3.Session:
    return aioboto3.Session()


def _client_kwargs() -> dict:
    settings = get_settings()
    return {
        "endpoint_url": settings.minio_endpoint_url,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
    }


async def ensure_bucket_exists() -> None:
    settings = get_settings()
    async with _session().client("s3", **_client_kwargs()) as s3:
        try:
            await s3.head_bucket(Bucket=settings.minio_bucket)
        except ClientError:
            await s3.create_bucket(Bucket=settings.minio_bucket)
            logger.info("storage_bucket_created", bucket=settings.minio_bucket)


async def upload_bytes(key: str, content: bytes) -> None:
    settings = get_settings()
    async with _session().client("s3", **_client_kwargs()) as s3:
        await s3.put_object(Bucket=settings.minio_bucket, Key=key, Body=content)


async def download_bytes(key: str) -> bytes:
    settings = get_settings()
    async with _session().client("s3", **_client_kwargs()) as s3:
        response = await s3.get_object(Bucket=settings.minio_bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()


async def check_storage_connection() -> bool:
    try:
        settings = get_settings()
        async with _session().client("s3", **_client_kwargs()) as s3:
            await s3.head_bucket(Bucket=settings.minio_bucket)
        return True
    except Exception:
        return False
