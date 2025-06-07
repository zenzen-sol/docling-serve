import argparse
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import torch
from pypdf import PdfReader

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_serve.http_logging import (
    HttpJsonLogHandler,
    PageProgressLogHandler,
)
from docling_serve.storage import download_from_gcs

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def send_progress_update(
    callback_url: Optional[str],
    token: Optional[str],
    source_id: str,
    status: str,
    message: str,
    error: Optional[str] = None,
    total_tokens_estimate: Optional[int] = None,
    processed_pages: Optional[int] = None,
    total_pages: Optional[int] = None,
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
        if total_tokens_estimate is not None:
            payload["total_tokens_estimate"] = total_tokens_estimate
        if processed_pages is not None:
            payload["processed_pages"] = processed_pages
        if total_pages is not None:
            payload["total_pages"] = total_pages

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


def run_conversion(
    source_url, source_id, callback_url, progress_callback_url, progress_callback_token
):
    """The main conversion process."""
    total_pages = 0
    total_tokens_estimate = None
    vlm_logger = None
    page_progress_handler = None
    file_stream = None

    try:
        # 1. Download the file to get page count and estimate tokens
        with tempfile.NamedTemporaryFile() as temp_pdf:
            download_from_gcs(source_url, temp_pdf.name)

            # Estimate total tokens based on page count
            try:
                pdf_reader = PdfReader(temp_pdf.name)
                total_pages = len(pdf_reader.pages)
                ESTIMATED_TOKENS_PER_PAGE = 2000
                total_tokens_estimate = total_pages * ESTIMATED_TOKENS_PER_PAGE
                _log.info(
                    f"Estimated token count for {total_pages} pages: {total_tokens_estimate}"
                )
            except Exception:
                _log.warning(
                    "Could not estimate token count. File may not be a standard PDF."
                )

            # Prepare the file stream for conversion
            temp_pdf.seek(0)
            file_stream = BytesIO(temp_pdf.read())

        # 2. Set up logging if a callback is provided
        if progress_callback_url:
            page_progress_handler = PageProgressLogHandler(
                url=progress_callback_url,
                source_id=source_id,
                total_pages=total_pages,
                token=progress_callback_token,
            )
            vlm_logger = logging.getLogger("docling.models.hf_vlm_model")
            vlm_logger.addHandler(page_progress_handler)
            vlm_logger.setLevel(logging.DEBUG)

        # 3. Send a starting progress update
        send_progress_update(
            callback_url=progress_callback_url,
            token=progress_callback_token,
            source_id=source_id,
            status="PROCESSING",
            message="Starting document conversion process.",
            total_tokens_estimate=total_tokens_estimate,
            total_pages=total_pages,
        )

        # 4. Set up the conversion options
        pipeline_options = VlmPipelineOptions(
            accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=True)
        )
        pipeline_options.vlm_options = smoldocling_vlm_conversion_options
        vlm_pipeline = VlmPipeline(pipeline_options)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_instance=vlm_pipeline
                )
            }
        )

        # 5. Prepare the source document.
        file_name = Path(source_url).name.split("?")[0]
        sources = [DocumentStream(name=file_name, stream=file_stream)]

        _log.info(f"Starting conversion for {file_name}...")

        # 6. Run the conversion.
        results = converter.convert_all(sources)
        result: ConversionResult = next(results)

        if result.document:
            _log.info(
                f"Conversion successful for {file_name}. Preparing to send result."
            )
            doc_json_str = result.document.model_dump_json()
            with BytesIO(doc_json_str.encode("utf-8")) as f:
                files = {"docling_file": ("docling.json", f, "application/json")}
                response = requests.post(callback_url, files=files)
            response.raise_for_status()
            _log.info(f"Callback successful with status code: {response.status_code}")
        elif result.error:
            raise result.error

    except Exception as e:
        _log.error(
            f"An unexpected error occurred during job execution: {e}", exc_info=True
        )
        send_progress_update(
            callback_url=progress_callback_url,
            token=progress_callback_token,
            source_id=source_id,
            status="FAILED",
            message="An unexpected error occurred during job execution.",
            error=str(e),
        )
        raise
    finally:
        if vlm_logger and page_progress_handler:
            vlm_logger.removeHandler(page_progress_handler)
        if file_stream:
            file_stream.close()


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
        progress_callback_url_with_id = f"{args.progress_callback_url}/{args.source_id}"
        http_handler = HttpJsonLogHandler(
            url=progress_callback_url_with_id, token=args.progress_callback_token
        )
        # We only want to see DEBUG messages from the 'docling' logger
        http_handler.setLevel(logging.DEBUG)
        logging.getLogger("docling").addHandler(http_handler)

    _log.info(f"Starting job for source_id: {args.source_id}")
    _log.info(f"  - Source URL: {args.source_url}")
    _log.info(f"  - Callback URL: {args.callback_url}")
    _log.info(f"  - Progress Callback URL: {progress_callback_url_with_id}")
    _log.info("=" * 50)

    try:
        run_conversion(
            args.source_url,
            args.source_id,
            args.callback_url,
            progress_callback_url_with_id,
            args.progress_callback_token,
        )
    except Exception as e:
        _log.error(f"Job failed for source_id: {args.source_id}", exc_info=True)
        # The exception is re-raised to ensure the Cloud Run Job fails correctly.
        raise

    _log.info("Job finished successfully.")


if __name__ == "__main__":
    main()
