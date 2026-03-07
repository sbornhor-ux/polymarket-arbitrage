"""
Cloudflare R2 Client (S3-compatible) using boto3.
---------------------------------------------------
Thin wrapper around boto3 for the S3-compatible Cloudflare R2 API.

Public API:
  upload_file(filepath, key)  — upload a local file; returns public URL
  list_objects(prefix)        — list bucket contents; returns list of dicts
  download_file(key, dest)    — download an object to a local path
  parse_r2_url(url)           — parse endpoint/bucket/prefix from an R2 URL

Credentials are read from environment variables (in order of preference):
  AWS_ACCESS_KEY_ID  / AWS_SECRET_ACCESS_KEY   (standard AWS convention)
  R2_ACCESS_KEY_ID   / R2_SECRET_ACCESS_KEY    (R2-specific aliases)

If credentials are absent, upload_file and download_file raise RuntimeError;
list_objects returns an empty list so callers can degrade gracefully.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Optional, List, Dict

import boto3
import botocore

from config import R2_BUCKET_URL


def parse_r2_url(r2_url: str) -> tuple[str, str, str]:
    """Return (endpoint_url, bucket, prefix) from an R2 URL like
    https://<account>.r2.cloudflarestorage.com/my-bucket[/prefix]
    """
    parsed = urlparse(r2_url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.lstrip("/")
    parts = path.split("/") if path else []
    bucket = parts[0] if parts else ""
    prefix = "/".join(parts[1:]) if len(parts) > 1 else ""
    return endpoint, bucket, prefix


def _get_s3_client() -> Optional[boto3.client]:
    """
    Build an S3 client pointed at the R2 endpoint.
    Returns None if credentials are not configured (callers should handle gracefully).
    """
    endpoint, _bucket, _ = parse_r2_url(R2_BUCKET_URL)
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
    token = os.getenv("AWS_SESSION_TOKEN")  # optional; not required for R2

    if not access_key or not secret:
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret,
        aws_session_token=token,
    )


def upload_file(filepath: str, key: Optional[str] = None) -> str:
    """Upload a local file to the R2 bucket. Returns the public URL (if possible).

    Raises RuntimeError if credentials are not configured or upload fails.
    """
    endpoint, bucket, prefix = parse_r2_url(R2_BUCKET_URL)
    client = _get_s3_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")

    key = key or os.path.basename(filepath)
    if prefix:
        key = f"{prefix.rstrip('/')}/{key}"

    try:
        client.upload_file(filepath, bucket, key)
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(f"R2 upload failed: {e}") from e

    # Construct URL to object (public access depends on bucket policy)
    url = f"{endpoint}/{bucket}/{key}"
    return url


def list_objects(prefix: str = "") -> List[Dict]:
    """List objects in the R2 bucket under optional prefix. Returns list of dicts
    with keys: Key, LastModified, Size.
    Returns empty list if R2 not configured.
    """
    client = _get_s3_client()
    if client is None:
        return []

    _, bucket, base_prefix = parse_r2_url(R2_BUCKET_URL)
    full_prefix = "/".join([p for p in [base_prefix, prefix] if p]).lstrip("/")
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
        contents = resp.get("Contents", [])
        return [{"Key": c["Key"], "LastModified": c["LastModified"], "Size": c["Size"]} for c in contents]
    except botocore.exceptions.ClientError:
        return []


def download_file(key: str, dest_path: str) -> None:
    """Download object `key` from R2 bucket to local `dest_path`.

    Raises RuntimeError if not configured or download fails.
    """
    client = _get_s3_client()
    if client is None:
        raise RuntimeError("R2 credentials not configured")

    _, bucket, _ = parse_r2_url(R2_BUCKET_URL)
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        client.download_file(bucket, key, dest_path)
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(f"R2 download failed: {e}") from e
