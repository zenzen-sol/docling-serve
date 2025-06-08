"""
GCS Model Cache - Runtime model loading from Google Cloud Storage
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GCSModelCache:
    """Handles downloading and caching models from GCS at runtime."""

    def __init__(
        self,
        bucket_name: str = "jarvis-444507-docling-models",
        local_cache_path: str = "/tmp/docling-models",
    ):
        self.bucket_name = bucket_name
        self.local_cache_path = Path(local_cache_path)
        self.models_path = self.local_cache_path / ".cache/docling/models"

    def ensure_models_available(self) -> str:
        """
        Ensure models are available locally, downloading from GCS if needed.
        Returns the path to the models directory.
        """
        if self.models_path.exists() and self._models_are_complete():
            logger.info(f"Models already cached at {self.models_path}")
            return str(self.models_path)

        logger.info("Downloading models from GCS...")
        self._download_models_from_gcs()

        if not self._models_are_complete():
            raise RuntimeError("Model download from GCS failed - models incomplete")

        logger.info(f"Models ready at {self.models_path}")
        return str(self.models_path)

    def _download_models_from_gcs(self) -> None:
        """Download models from GCS bucket to local cache."""
        try:
            # Create local directory
            self.local_cache_path.mkdir(parents=True, exist_ok=True)

            # Download from GCS
            gcs_source = f"gs://{self.bucket_name}/models/"
            local_target = str(self.models_path)

            cmd = ["gsutil", "-m", "cp", "-r", gcs_source, local_target]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            logger.info("GCS download completed successfully")
            if result.stdout:
                logger.debug(f"gsutil stdout: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download models from GCS: {e}")
            if e.stderr:
                logger.error(f"gsutil stderr: {e.stderr}")
            raise RuntimeError(f"GCS model download failed: {e}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("GCS model download timed out after 5 minutes")

    def _models_are_complete(self) -> bool:
        """Check if all expected models are present locally."""
        if not self.models_path.exists():
            return False

        # Basic check - models directory should have some content
        try:
            model_files = list(self.models_path.rglob("*"))
            has_content = len(model_files) > 10  # Expect many model files

            if has_content:
                logger.debug(f"Found {len(model_files)} model files")
            else:
                logger.warning(
                    f"Only found {len(model_files)} model files - may be incomplete"
                )

            return has_content
        except Exception as e:
            logger.error(f"Error checking model completeness: {e}")
            return False


def setup_gcs_models(
    bucket_name: Optional[str] = None, local_cache_path: Optional[str] = None
) -> str:
    """
    Convenience function to set up GCS model caching.
    Returns the local path where models are available.
    """
    cache = GCSModelCache(
        bucket_name=bucket_name or "jarvis-444507-docling-models",
        local_cache_path=local_cache_path or "/tmp/docling-models",
    )

    models_path = cache.ensure_models_available()

    # Set environment variable for Docling to find models
    os.environ["DOCLING_SERVE_ARTIFACTS_PATH"] = models_path
    logger.info(f"Set DOCLING_SERVE_ARTIFACTS_PATH={models_path}")

    return models_path
