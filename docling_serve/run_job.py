import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests
import torch

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_serve.datamodel.convert import ConvertDocumentsOptions
from docling_serve.docling_conversion import get_converter, get_pdf_pipeline_opts

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


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


def _progress_callback(event: str, message: str, **kwargs: Any) -> None:
    """A simple callback to log progress events."""
    # current time with milliseconds
    current_time = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        + f".{int(time.time() * 1000) % 1000:03d}"
    )
    _log.info(f"[{current_time}] PIPELINE_EVENT: {event} - {message} {kwargs}")


def main():
    """
    Main entry point for the docling-serve job.
    This script is intended to be run as a Cloud Run Job.
    """
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
    # We can add more arguments from ConvertDocumentsOptions later if needed
    # For now, we will use the defaults for the VLM pipeline.

    args = parser.parse_args()

    _log.info("Starting docling-serve job with the following arguments:")
    _log.info(f"  Source URL: {args.source_url}")
    _log.info(f"  Callback URL: {args.callback_url}")
    _log.info(f"  Source ID: {args.source_id}")

    # TODO: Add logic to update the database status to "Processing"
    # This will require database credentials to be securely provided to the job.
    _log.info(f"Updating source {args.source_id} status to 'Processing' (placeholder).")

    try:
        # 1. Set up the conversion options.
        # We are hardcoding the VLM pipeline options for now.
        # This can be made configurable with more command-line arguments if needed.
        options = ConvertDocumentsOptions(pipeline="vlm")
        pdf_format_option: PdfFormatOption = get_pdf_pipeline_opts(options)

        # We need to manually construct the converter to inject our callback.
        # The `get_converter` helper doesn't support this directly.
        pdf_format_option.pipeline_cls = VlmPipeline
        pdf_format_option.pipeline_options.progress_callback = _progress_callback

        converter = DocumentConverter(pdf_format_option=pdf_format_option)

        # 2. Prepare the source document.
        # The 'name' can be extracted from the URL if necessary, but is not critical.
        file_name = Path(args.source_url).name.split("?")[0]
        sources = [DocumentStream(name=file_name, url=args.source_url)]

        _log.info(f"Starting conversion for {file_name}...")

        # 3. Run the conversion. This is the main, long-running step.
        results = converter.convert_all(sources)
        result: ConversionResult = next(results)

        if result.document:
            _log.info(f"Conversion successful for {file_name}.")
            document_json = result.document.model_dump_json()

            # 4. POST the result to the callback URL.
            _log.info(f"Sending result to callback URL: {args.callback_url}")
            # In a real scenario, you'd want more robust error handling here.
            response = requests.post(
                args.callback_url,
                data=document_json,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            _log.info(f"Callback successful with status code: {response.status_code}")

        elif result.error:
            _log.error(f"Conversion failed for {file_name}: {result.error}")
            # TODO: Add logic to update the database status to "Failed".
            raise result.error

    except Exception as e:
        _log.error(
            f"An unexpected error occurred during job execution: {e}", exc_info=True
        )
        # TODO: Add logic to update the database status to "Failed".
        raise

    _log.info("Job finished successfully.")


if __name__ == "__main__":
    main()
