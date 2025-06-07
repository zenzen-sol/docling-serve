import argparse
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
import torch

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_serve.datamodel.convert import ConvertDocumentsOptions
from docling_serve.docling_conversion import get_converter, get_pdf_pipeline_opts
from docling_serve.http_logging import HttpJsonLogHandler

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def send_progress_update(
    callback_url: Optional[str],
    token: Optional[str],
    source_id: str,
    status: str,
    message: str,
    error: Optional[str] = None,
):
    """Sends a progress update to the provided callback URL."""
    if not callback_url:
        _log.warning("No progress callback URL provided. Skipping update.")
        return

    _log.info(f"Sending progress update: status='{status}', message='{message}'")
    try:
        payload = {"status": status, "message": message, "source_id": source_id}
        if error:
            payload["error"] = error

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            callback_url, json=payload, headers=headers, timeout=15
        )
        response.raise_for_status()
        _log.info("Progress update sent successfully.")
    except requests.RequestException as e:
        _log.error(f"Failed to send progress update: {e}", exc_info=True)


def log_gpu_diagnostics():
    """Logs detailed information about the CUDA environment."""
    _log.info("=" * 50)
    _log.info("GPU DIAGNOSTICS")
    _log.info("=" * 50)
    if torch.cuda.is_available():
        _log.info("✅ CUDA is available.")
        device_count = torch.cuda.device_count()
        _log.info(f"   - Found {device_count} CUDA device(s).")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            _log.info(f"   - Device {i}: {device_name}")
    else:
        _log.error("❌ CUDA is NOT available. Processing will be VERY slow.")
    _log.info("=" * 50)


def main():
    """
    Main entry point for the docling-serve job.
    This script is intended to be run as a Cloud Run Job.
    """
    # Set the docling logger to DEBUG to get more granular progress.
    logging.getLogger("docling").setLevel(logging.DEBUG)

    # Run diagnostics first
    log_gpu_diagnostics()

    parser = argparse.ArgumentParser(
        description="Process a single document with docling."
    )
    parser.add_argument(
        "--source-url",
        type=str,
        required=True,
        help="The signed URL of the source document to process.",
    )
    parser.add_argument(
        "--callback-url",
        type=str,
        required=True,
        help="The callback URL to POST the final result to.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        required=True,
        help="The source_id from the database for status updates.",
    )
    parser.add_argument(
        "--progress-callback-url",
        type=str,
        required=False,
        help="The callback URL to POST progress updates to.",
    )
    parser.add_argument(
        "--progress-callback-token",
        type=str,
        required=False,
        help="A shared secret to authenticate progress callbacks.",
    )
    # We can add more arguments from ConvertDocumentsOptions later if needed
    # For now, we will use the defaults for the VLM pipeline.

    args = parser.parse_args()

    # If a progress callback URL is provided, set up the HTTP logger
    if args.progress_callback_url:
        # Note: We do not append the source_id here, as the handler will
        # be used for generic log messages. The receiving API should handle
        # associating the logs with the job via the source_id in the payload.
        # However, for simplicity in this implementation, we will pass the full URL
        # and assume the API can handle it.
        # A more robust solution might involve a separate endpoint for logs.
        progress_callback_url_with_id = (
            f"{args.progress_callback_url.rstrip('/')}/{args.source_id}"
        )
        http_handler = HttpJsonLogHandler(
            url=progress_callback_url_with_id, token=args.progress_callback_token
        )
        # We only want to see DEBUG messages from the 'docling' logger
        http_handler.setLevel(logging.DEBUG)
        logging.getLogger("docling").addHandler(http_handler)

    _log.info("Starting docling-serve job with the following arguments:")
    _log.info(f"  Source URL: {args.source_url}")
    _log.info(f"  Callback URL: {args.callback_url}")
    _log.info(f"  Source ID: {args.source_id}")
    _log.info(f"  Progress Callback URL: {args.progress_callback_url}")

    progress_callback_url_with_id = (
        f"{args.progress_callback_url.rstrip('/')}/{args.source_id}"
        if args.progress_callback_url
        else None
    )

    try:
        send_progress_update(
            callback_url=progress_callback_url_with_id,
            token=args.progress_callback_token,
            source_id=args.source_id,
            status="PROCESSING",
            message="Conversion job status: [WORKING]",
        )

        # 1. Set up the conversion options.
        # Create VLM pipeline options directly to work around a bug in DocumentConverter.
        # This ensures the VLM pipeline is actually used.
        pipeline_options = VlmPipelineOptions(
            accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=True)
        )
        pipeline_options.vlm_options = smoldocling_vlm_conversion_options

        # Instantiate the VLM Pipeline and DocumentConverter directly.
        vlm_pipeline = VlmPipeline(pipeline_options)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_instance=vlm_pipeline
                )
            }
        )

        # 2. Prepare the source document.
        file_name = Path(args.source_url).name.split("?")[0]

        _log.info(f"Downloading source file from URL for {file_name}...")
        download_response = requests.get(args.source_url)
        download_response.raise_for_status()  # Ensure the download was successful
        file_content = download_response.content
        _log.info(f"Successfully downloaded {len(file_content)} bytes.")

        sources = [DocumentStream(name=file_name, stream=BytesIO(file_content))]

        _log.info(f"Starting conversion for {file_name}...")

        # 3. Run the conversion. This is the main, long-running step.
        results = converter.convert_all(sources)
        result: ConversionResult = next(results)

        if result.document:
            _log.info(
                f"Conversion successful for {file_name}. Preparing to send result as a file."
            )

            # Use the Pydantic model's own .model_dump_json() method. This correctly
            # serializes special Pydantic types (like AnyUrl) into a valid JSON string.
            doc_json_str = result.document.model_dump_json()

            # Create an in-memory binary stream to send as a file. This is the robust
            # way to handle potentially large results and avoid API payload size limits.
            with BytesIO(doc_json_str.encode("utf-8")) as f:
                files = {"docling_file": ("docling.json", f, "application/json")}
                _log.info(f"Sending result file to callback URL: {args.callback_url}")
                response = requests.post(args.callback_url, files=files)

            response.raise_for_status()
            _log.info(f"Callback successful with status code: {response.status_code}")

        elif result.error:
            _log.error(f"Conversion failed for {file_name}: {result.error}")
            # This is now handled by the exception handler below.
            raise result.error

    except Exception as e:
        _log.error(
            f"An unexpected error occurred during job execution: {e}", exc_info=True
        )
        send_progress_update(
            callback_url=progress_callback_url_with_id,
            token=args.progress_callback_token,
            source_id=args.source_id,
            status="FAILED",
            message="An unexpected error occurred during job execution.",
            error=str(e),
        )
        # This is now handled by a callback to the API.
        raise

    _log.info("Job finished successfully.")


if __name__ == "__main__":
    main()
