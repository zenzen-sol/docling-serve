import tempfile
from functools import lru_cache
from pathlib import Path

import requests

from docling_serve.settings import docling_serve_settings


@lru_cache
def get_scratch() -> Path:
    scratch_dir = (
        docling_serve_settings.scratch_path
        if docling_serve_settings.scratch_path is not None
        else Path(tempfile.mkdtemp(prefix="docling_"))
    )
    scratch_dir.mkdir(exist_ok=True, parents=True)
    return scratch_dir


def download_from_gcs(url: str, output_path: str) -> None:
    """
    Download a file from a URL (typically a GCS signed URL) to a local path.

    Args:
        url: The URL to download from (can be GCS signed URL or any HTTP URL)
        output_path: Local file path to save the downloaded content
    """
    response = requests.get(url, timeout=300)  # 5 minute timeout for large files
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)
