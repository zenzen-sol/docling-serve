import argparse
import logging
import os
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import torch
from pypdf import PdfReader

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_serve.datamodel.convert import ConvertDocumentsOptions
from docling_serve.docling_conversion import _should_use_vllm_batching
from docling_serve.http_logging import (
    HttpJsonLogHandler,
    PageProgressLogHandler,
)
from docling_serve.storage import download_from_gcs

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# VlmResultCollector is no longer needed with current pipeline architecture

# Suppress transformers noise early - before any model loading
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*truncation.*")
warnings.filterwarnings("ignore", message=".*max_length.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")

# Set transformers logging level
try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass

# Global tracking for progress
start_time = None
total_pages_global = 0


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
    elapsed_minutes: Optional[float] = None,
    estimated_remaining_minutes: Optional[float] = None,
):
    """Sends a progress update to the provided callback URL."""
    if not callback_url:
        _log.warning("No progress callback URL provided. Skipping update.")
        return

    # Calculate percentage if we have the data
    percentage = None
    if processed_pages is not None and total_pages is not None and total_pages > 0:
        percentage = round((processed_pages / total_pages) * 100, 1)

    progress_msg = message
    if processed_pages is not None and total_pages is not None:
        progress_msg = f"Page {processed_pages}/{total_pages}"
        if percentage is not None:
            progress_msg += f" ({percentage}%)"
        if elapsed_minutes is not None:
            progress_msg += f" - {elapsed_minutes:.1f}m elapsed"
        if estimated_remaining_minutes is not None:
            progress_msg += f", ~{estimated_remaining_minutes:.1f}m remaining"

    _log.info(f"📈 Progress: {progress_msg}")

    try:
        payload = {"status": status, "message": progress_msg, "source_id": source_id}
        if error:
            payload["error"] = error
        if total_tokens_estimate is not None:
            payload["total_tokens_estimate"] = total_tokens_estimate
        if processed_pages is not None:
            payload["processed_pages"] = processed_pages
        if total_pages is not None:
            payload["total_pages"] = total_pages
        if percentage is not None:
            payload["percentage"] = percentage
        if elapsed_minutes is not None:
            payload["elapsed_minutes"] = elapsed_minutes
        if estimated_remaining_minutes is not None:
            payload["estimated_remaining_minutes"] = estimated_remaining_minutes

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            callback_url, json=payload, headers=headers, timeout=15
        )
        response.raise_for_status()
        _log.info("✅ Progress update sent successfully to maboroshi-api")
    except requests.RequestException as e:
        _log.error(
            f"❌ Failed to send progress update to maboroshi-api: {e}", exc_info=True
        )


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

    # Check flash-attention installation
    try:
        import flash_attn

        _log.info(f"✅ Flash-attention installed: {flash_attn.__version__}")
    except ImportError:
        _log.error("❌ Flash-attention NOT installed")

    # Check relevant environment variables
    _log.info("🔧 Environment variables:")
    flash_env = os.environ.get("FLASH_ATTENTION_SKIP_CUDA_BUILD", "not set")
    _log.info(f"   - FLASH_ATTENTION_SKIP_CUDA_BUILD: {flash_env}")
    transformers_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "not set")
    _log.info(f"   - TRANSFORMERS_VERBOSITY: {transformers_verbosity}")
    _log.info("=" * 50)


def run_conversion(
    source_url, source_id, callback_url, progress_callback_url, progress_callback_token
):
    """The main conversion process."""
    global start_time, total_pages_global

    total_pages = 0
    total_tokens_estimate = None
    vlm_logger = None
    page_progress_handler = None
    file_stream = None
    start_time = time.time()

    try:
        # Enable GCS model caching (if configured)
        if os.environ.get("USE_GCS_MODELS", "").lower() == "true":
            from docling_serve.gcs_model_cache import setup_gcs_models

            setup_gcs_models()
            _log.info("GCS model caching initialized")

        # 1. Download the file to get page count and estimate tokens
        with tempfile.NamedTemporaryFile() as temp_pdf:
            download_from_gcs(source_url, temp_pdf.name)

            # Estimate total tokens based on page count
            try:
                pdf_reader = PdfReader(temp_pdf.name)
                total_pages = len(pdf_reader.pages)
                total_pages_global = total_pages
                ESTIMATED_TOKENS_PER_PAGE = 2000
                total_tokens_estimate = total_pages * ESTIMATED_TOKENS_PER_PAGE
                _log.info(
                    f"📄 Document has {total_pages} pages (estimated {total_tokens_estimate} tokens)"
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
            # Create enhanced progress handler
            page_progress_handler = EnhancedPageProgressLogHandler(
                url=progress_callback_url,
                source_id=source_id,
                total_pages=total_pages,
                token=progress_callback_token,
            )
            # Attach to both the vLLM pipeline logger and the standard VLM logger for compatibility
            vllm_logger = logging.getLogger("docling_serve.vllm_vlm_pipeline")
            vlm_logger = logging.getLogger(
                "docling.models.vlm_models_inline.hf_transformers_model"
            )

            vllm_logger.addHandler(page_progress_handler)
            vlm_logger.addHandler(page_progress_handler)
            vllm_logger.setLevel(logging.INFO)  # Ensure we capture INFO level messages
            vlm_logger.setLevel(logging.DEBUG)

            _log.info(
                f"🔗 Progress tracking attached to loggers: {vllm_logger.name}, {vlm_logger.name}"
            )

        # 3. Send a starting progress update
        send_progress_update(
            callback_url=progress_callback_url,
            token=progress_callback_token,
            source_id=source_id,
            status="PROCESSING",
            message=f"Starting conversion of {total_pages} pages",
            total_tokens_estimate=total_tokens_estimate,
            total_pages=total_pages,
            processed_pages=0,
            elapsed_minutes=0.0,
        )

        # 4. Set up the conversion options
        _log.info("🔧 Configuring VLM pipeline with flash-attention...")

        # Create custom VLM options optimized for flash-attention
        # The key change: disable 8-bit quantization which conflicts with flash-attention
        # Use model_copy to safely create a modified version of the default options

        custom_vlm_options = smoldocling_vlm_conversion_options.model_copy(
            update={
                "load_in_8bit": False
            }  # 🔧 Disable 8-bit quantization for flash-attention compatibility
        )

        # Debug: Compare original vs custom options
        _log.info(
            f"🔍 Original options load_in_8bit: {smoldocling_vlm_conversion_options.load_in_8bit}"
        )
        _log.info(f"🔍 Custom options load_in_8bit: {custom_vlm_options.load_in_8bit}")
        _log.info(f"🔍 Custom options dump: {custom_vlm_options.model_dump()}")

        # A. Create a dummy request object for the check function
        dummy_request = ConvertDocumentsOptions()

        # B. Decide which pipeline class to use
        try:
            from docling_serve.vllm_vlm_pipeline import VllmBatchVlmPipeline

            if _should_use_vllm_batching(dummy_request):
                _log.info("🚀 Using vLLM batching pipeline for enhanced performance")
                pipeline_class = VllmBatchVlmPipeline
            else:
                _log.info("📝 Using standard VLM pipeline")
                pipeline_class = VlmPipeline
        except ImportError:
            _log.error(
                "Could not import VllmBatchVlmPipeline, defaulting to VlmPipeline"
            )
            pipeline_class = VlmPipeline

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_class,
                pipeline_options=VlmPipelineOptions(
                    # Enable FlashAttention for a significant performance boost on the GPU.
                    accelerator_options=AcceleratorOptions(
                        cuda_use_flash_attention2=True
                    ),
                    vlm_options=custom_vlm_options,
                ),
            )
        }
        _log.info(f"✅ Pipeline class selected: {pipeline_class.__name__}")

        # Note: VlmResultCollector is not compatible with current pipeline architecture
        # Diagnostic logging will rely on the standard ConversionResult object
        result_collector = None

        # 5. Run the conversion
        converter = DocumentConverter(
            format_options={InputFormat.PDF: format_options[InputFormat.PDF]}
        )
        result: ConversionResult = converter.convert(
            DocumentStream(stream=file_stream, name=source_url)
        )

        # --- DIAGNOSTIC LOGGING ---
        if result.document and result.document.pages:
            try:
                # Log conversion result summary for analysis
                page_count = len(result.document.pages)
                text_items = len(result.document.texts)
                _log.info(
                    f"🕵️‍♂️ Conversion Summary: {page_count} pages processed, "
                    f"{text_items} text items extracted"
                )
            except Exception as e:
                _log.error(f"❌ Failed to log conversion summary: {e}")
        # --- END DIAGNOSTIC LOGGING ---

        # Check conversion status and handle accordingly
        _log.info(f"Conversion status: {result.status}")

        if result.document and result.status in [
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        ]:
            elapsed_time = time.time() - start_time

            # Calculate performance metrics
            if total_pages > 0:
                pages_per_minute = total_pages / (elapsed_time / 60)
                _log.info(f"📊 Performance: {pages_per_minute:.1f} pages/minute")

            _log.info(
                f"✅ Conversion successful for {source_url} in {elapsed_time:.2f} seconds. Preparing to send result."
            )

            # 6. Send the final result to the main callback URL as a direct JSON payload.
            _log.info(
                "Fingerprint: Preparing to send direct JSON payload with definitive fix."
            )
            doc_json_str = result.document.model_dump_json()
            headers = {"Content-Type": "application/json"}
            if progress_callback_token:
                headers["Authorization"] = f"Bearer {progress_callback_token}"

            response = requests.post(callback_url, data=doc_json_str, headers=headers)
            response.raise_for_status()
            _log.info(
                f"✅ Callback successful with status code: {response.status_code}"
            )
        else:
            # Handle failure cases
            error_msg = f"Conversion failed with status: {result.status}"
            if result.errors:
                error_msg += f", errors: {', '.join(str(e) for e in result.errors)}"
            raise Exception(error_msg)

    except Exception as e:
        elapsed_time = time.time() - start_time if start_time else 0
        _log.error(
            f"❌ Job failed after {elapsed_time / 60:.1f} minutes: {e}", exc_info=True
        )
        send_progress_update(
            callback_url=progress_callback_url,
            token=progress_callback_token,
            source_id=source_id,
            status="FAILED",
            message=f"Job failed after {elapsed_time / 60:.1f} minutes",
            error=str(e),
        )
        raise
    finally:
        # Clean up progress handlers from both loggers
        if progress_callback_url and "page_progress_handler" in locals():
            try:
                vllm_logger = logging.getLogger("docling_serve.vllm_vlm_pipeline")
                vlm_logger = logging.getLogger(
                    "docling.models.vlm_models_inline.hf_transformers_model"
                )
                vllm_logger.removeHandler(page_progress_handler)
                vlm_logger.removeHandler(page_progress_handler)
            except Exception as cleanup_e:
                _log.warning(f"Could not clean up progress handlers: {cleanup_e}")
        if file_stream:
            file_stream.close()


class EnhancedPageProgressLogHandler(logging.Handler):
    """Enhanced progress handler with timing and percentage calculations for vLLM batching."""

    def __init__(
        self, url: str, source_id: str, total_pages: int, token: Optional[str] = None
    ):
        super().__init__()
        self.url = url
        self.source_id = source_id
        self.total_pages = total_pages
        self.token = token
        self.processed_pages = 0

        # Updated regex patterns to match vLLM batch processing messages
        # Match both individual page completion and batch completion messages
        self.progress_patterns = [
            re.compile(
                r"Page \d+: DocTags stored \(\d+ chars\)"
            ),  # Individual page completion
            re.compile(
                r"✅ Batch #\d+:.*?(\d+\.\d+)s/page"
            ),  # Batch completion with timing
            re.compile(
                r"Generated \d+ tokens"
            ),  # Fallback for standard HF transformers
        ]

    def emit(self, record: logging.LogRecord):
        """Enhanced progress tracking with timing estimates for vLLM batching."""
        message = record.getMessage()

        # Check if this is a progress-related message
        pages_in_message = 0
        for pattern in self.progress_patterns:
            if pattern.search(message):
                # Individual page completion - count as 1 page
                if "Page" in message and "DocTags stored" in message:
                    pages_in_message = 1
                elif "✅ Batch #" in message and "s/page" in message:
                    # Batch completion - extract page count from the preceding batch start message
                    # We'll count this as 0 since individual pages were already counted
                    pages_in_message = 0
                else:
                    # Standard HF transformers message - count as 1 page
                    pages_in_message = 1
                break

        if pages_in_message > 0:
            self.processed_pages += pages_in_message

            # Calculate timing
            elapsed_time = time.time() - start_time if start_time else 0
            elapsed_minutes = elapsed_time / 60

            # Estimate remaining time
            estimated_remaining_minutes = None
            if self.processed_pages > 0:
                avg_time_per_page = elapsed_time / self.processed_pages
                remaining_pages = self.total_pages - self.processed_pages
                estimated_remaining_minutes = (remaining_pages * avg_time_per_page) / 60

            # Send enhanced progress update
            send_progress_update(
                callback_url=self.url,
                token=self.token,
                source_id=self.source_id,
                status="PROCESSING",
                message=f"Processed page {self.processed_pages} of {self.total_pages}",
                processed_pages=self.processed_pages,
                total_pages=self.total_pages,
                elapsed_minutes=elapsed_minutes,
                estimated_remaining_minutes=estimated_remaining_minutes,
            )


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

    if not args.source_url:
        parser.error("Either --source-url is required.")

    # If a progress callback URL is provided, set up the HTTP logger
    progress_callback_url_with_id = None
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
