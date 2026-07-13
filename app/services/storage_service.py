"""S3 파일 업로드 및 삭제를 담당한다."""

from pathlib import Path

import boto3

from app.core.config import settings


s3_client = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
)


# 로컬 임시 파일을 S3에 업로드한다.
def upload_file(local_path: str, object_key: str) -> str:
    path = Path(local_path)

    if not path.exists():
        raise FileNotFoundError(f"업로드할 파일이 없습니다: {local_path}")

    s3_client.upload_file(
        Filename=str(path),
        Bucket=settings.S3_BUCKET_NAME,
        Key=object_key,
    )

    return object_key


# S3에 저장된 파일을 삭제한다.
def delete_file(object_key: str) -> None:
    if not object_key:
        return

    s3_client.delete_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=object_key,
    )


# S3 객체 키를 생성한다.
def build_object_key(
    user_id: str,
    session_id: str,
    filename: str,
) -> str:
    prefix = settings.S3_UPLOAD_PREFIX.strip("/")

    return f"{prefix}/{user_id}/{session_id}/{filename}"