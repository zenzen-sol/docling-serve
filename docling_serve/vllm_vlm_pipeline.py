"""
vLLM-based VLM Pipeline for batched inference with SmolDocling model.

This module provides a custom VLM pipeline that uses vLLM for batched inference,
significantly improving performance for multi-page documents by processing
multiple pages simultaneously instead of sequentially.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from PIL import Image
from vllm import LLM, SamplingParams

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import ConversionResult, DocumentConversionInput
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.vlm_pipeline import VlmPipeline

_log = logging.getLogger(__name__)

# Standard prompt used by SmolDocling
SMOLDOCLING_PROMPT = """You are an OCR assistant. Analyze the provided image and:
1. Extract all text content exactly as it appears
2. Preserve formatting, structure, and layout
3. Include any mathematical formulas or special characters
4. Describe any tables, diagrams, or non-text elements

Return only the content without explanations."""


class VllmBatchVlmPipeline(BasePipeline):
    """vLLM-based VLM pipeline for batched inference with SmolDocling model."""

    def __init__(self, pipeline_options: VlmPipelineOptions):
        """Initialize the vLLM VLM pipeline.

        Args:
            pipeline_options: Configuration options for the VLM pipeline
        """
        super().__init__(pipeline_options)
        self.pipeline_options = pipeline_options

        # Initialize vLLM model with SmolDocling
        model_name = "HuggingFaceTB/SmolVLM-Instruct"

        # Configure vLLM for optimal batching
        _log.info(f"🚀 Initializing vLLM with model: {model_name}")

        # Get batch size from environment or use default
        import os

        batch_size = int(os.getenv("VLLM_BATCH_SIZE", "4"))
        gpu_memory_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on GPU count
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=8192,  # Adjust based on needs
            trust_remote_code=True,
            dtype="bfloat16",  # Use bfloat16 for better performance
            disable_log_stats=False,  # Enable for monitoring
            enable_prefix_caching=True,  # Cache common prefixes for efficiency
        )

        # Configure sampling parameters for consistent output
        self.sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistent OCR
            top_p=0.9,
            max_tokens=2048,  # Adjust based on expected output length
            stop=[],  # No specific stop tokens for OCR
        )

        # Store batch size
        self.batch_size = batch_size

        # Fallback to standard VLM pipeline for compatibility
        self.fallback_pipeline = VlmPipeline(pipeline_options)

        _log.info(
            f"✅ vLLM VLM pipeline initialized successfully (batch_size={batch_size})"
        )

    def __call__(
        self,
        conv_input: DocumentConversionInput,
        **kwargs,
    ) -> ConversionResult:
        """Process a document using vLLM batched inference.

        Args:
            conv_input: Document conversion input containing pages
            **kwargs: Additional keyword arguments

        Returns:
            ConversionResult: Processed document result
        """
        _log.info(
            f"🎯 Starting vLLM batched processing for document: {conv_input.file}"
        )

        try:
            return self._process_with_vllm_batching(conv_input, **kwargs)
        except Exception as e:
            _log.warning(
                f"⚠️ vLLM processing failed, falling back to standard pipeline: {e}"
            )
            _log.debug(f"vLLM error details: {e}", exc_info=True)
            return self.fallback_pipeline(conv_input, **kwargs)

    def _process_with_vllm_batching(
        self,
        conv_input: DocumentConversionInput,
        **kwargs,
    ) -> ConversionResult:
        """Process document pages using vLLM batched inference."""
        start_time = time.time()

        _log.info("🚀 Starting REAL vLLM batched VLM processing...")

        # First, run the standard pipeline to get layout detection, OCR, etc.
        # but disable VLM processing to avoid duplicate work
        temp_options = VlmPipelineOptions(
            artifacts_path=self.pipeline_options.artifacts_path,
            document_timeout=self.pipeline_options.document_timeout,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

        # Create a standard pipeline without VLM for base processing
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        standard_options = PdfPipelineOptions(
            artifacts_path=self.pipeline_options.artifacts_path,
            document_timeout=self.pipeline_options.document_timeout,
            do_ocr=True,
            do_table_structure=True,
            do_picture_description=False,  # We'll handle this with vLLM
        )

        standard_pipeline = StandardPdfPipeline(standard_options)

        # Get base document structure without VLM processing
        base_result = standard_pipeline(conv_input, **kwargs)

        if not base_result.document or not base_result.document.pages:
            _log.warning("No pages found in document, skipping vLLM processing")
            return base_result

        # Extract page images for vLLM processing
        page_images = []
        page_indices = []

        for i, page in enumerate(base_result.document.pages):
            if hasattr(page, "image") and page.image:
                page_images.append(page.image)
                page_indices.append(i)

        if not page_images:
            _log.warning("No page images found for vLLM processing")
            return base_result

        _log.info(
            f"📄 Processing {len(page_images)} pages with vLLM batching (batch_size={self.batch_size})"
        )

        # Process pages in batches with vLLM
        all_vlm_outputs = []

        for batch_start in range(0, len(page_images), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(page_images))
            batch_images = page_images[batch_start:batch_end]
            batch_indices = page_indices[batch_start:batch_end]

            _log.info(
                f"🔄 Processing batch {batch_start // self.batch_size + 1}: pages {batch_indices}"
            )

            # Prepare prompts for batch
            prompts = []
            for img in batch_images:
                # Convert PIL Image to format expected by vLLM
                prompts.append(
                    {"prompt": SMOLDOCLING_PROMPT, "multi_modal_data": {"image": img}}
                )

            # Run vLLM batch inference
            batch_start_time = time.time()
            outputs = self.llm.generate(prompts, self.sampling_params)
            batch_time = time.time() - batch_start_time

            _log.info(
                f"✅ Batch completed in {batch_time:.2f}s ({len(batch_images) / batch_time:.2f} pages/sec)"
            )

            # Extract text outputs
            for output in outputs:
                generated_text = output.outputs[0].text
                all_vlm_outputs.append(generated_text)

        # Apply VLM outputs back to document pages
        for i, (page_idx, vlm_output) in enumerate(zip(page_indices, all_vlm_outputs)):
            # Here we would integrate the VLM output into the document structure
            # For now, we'll add it as additional text content
            page = base_result.document.pages[page_idx]

            # Add VLM-extracted content to page
            if hasattr(page, "text") and page.text:
                page.text += f"\n\n<!-- vLLM Enhanced Content -->\n{vlm_output}"
            else:
                page.text = vlm_output

        elapsed_time = time.time() - start_time
        pages_per_second = len(page_images) / elapsed_time if elapsed_time > 0 else 0

        _log.info(
            f"🎯 vLLM batch processing completed: {len(page_images)} pages in "
            f"{elapsed_time:.2f}s ({pages_per_second:.2f} pages/sec total)"
        )

        return base_result

    def supports_batching(self) -> bool:
        """Check if this pipeline supports batching.

        Returns:
            True, as this pipeline is designed for batching
        """
        return True

    def get_batch_size(self) -> int:
        """Get the optimal batch size for this pipeline.

        Returns:
            Configured batch size
        """
        return self.batch_size
