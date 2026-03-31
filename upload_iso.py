"""Upload Zernel ISOs to Cloudflare R2 using S3-compatible API."""
import boto3
import sys
import os

# You need to create an R2 API token at:
# https://dash.cloudflare.com -> R2 -> Manage R2 API Tokens -> Create API Token
# Permissions: Object Read & Write, Bucket: zernel-iso

ACCOUNT_ID = "da8f8938ae6093b1835d0ae2be83c64e"
ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET = "zernel-iso"

# Set these from your R2 API token
ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")

if not ACCESS_KEY or not SECRET_KEY:
    print("Set your R2 API token credentials:")
    print("  $env:R2_ACCESS_KEY = 'your-access-key-id'")
    print("  $env:R2_SECRET_KEY = 'your-secret-access-key'")
    print()
    print("Get these from: https://dash.cloudflare.com -> R2 -> Manage R2 API Tokens")
    sys.exit(1)

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="auto",
)

files = [
    r"H:\zernel\output\zernel-0.1.0-server-amd64.iso",
    r"H:\zernel\output\zernel-0.1.0-desktop-amd64.iso",
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"SKIP: {filepath} not found")
        continue

    filename = os.path.basename(filepath)
    size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"Uploading {filename} ({size_gb:.1f} GB)...")

    s3.upload_file(
        filepath,
        BUCKET,
        filename,
        Callback=lambda bytes_transferred: None,
        Config=boto3.s3.transfer.TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100MB
            multipart_chunksize=100 * 1024 * 1024,
            max_concurrency=4,
        ),
    )
    print(f"  Done: https://zernel.org/{filename}")

print()
print("All uploads complete.")
