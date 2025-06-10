"""
vLLM-based VLM Pipeline for batched inference with SmolDocling model.

This module provides a VLM pipeline that uses vLLM for batched inference,
following the docling pipeline architecture by extending PaginatedPipeline
and using the build_pipe pattern.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
)
from docling.datamodel.settings import settings
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class VllmBatchVlmModel:
    """vLLM-based VLM model for batched inference."""

    def __init__(
        self,
        enabled: bool = True,
        artifacts_path: Optional[Path] = None,
        accelerator_options=None,
        vlm_options: Optional[InlineVlmOptions] = None,
    ):
        self.enabled = enabled
        self.artifacts_path = artifacts_path
        self.accelerator_options = accelerator_options
        self.vlm_options = vlm_options

        if enabled:
            self._initialize_vllm()

    def _initialize_vllm(self):
        """Initialize vLLM model."""
        try:
            from vllm import LLM, SamplingParams

            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            batch_size = int(os.getenv("VLLM_BATCH_SIZE", "4"))
            gpu_memory_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))

            _log.info(f"🚀 Initializing vLLM with model: {model_name}")

            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=8192,
                trust_remote_code=True,
                dtype="bfloat16",
                disable_log_stats=False,
                enable_prefix_caching=True,
            )

            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048,
                stop=[],
            )

            self.batch_size = batch_size
            _log.info(f"✅ vLLM initialized successfully (batch_size={batch_size})")

        except Exception as e:
            _log.error(f"❌ Failed to initialize vLLM: {e}")
            self.enabled = False
            raise

    def __call__(self, conv_res: ConversionResult, page_batch):
        """Process a batch of pages with vLLM."""
        if not self.enabled:
            return page_batch

        _log.info(f"🔄 Processing page batch with vLLM...")

        # Convert pages to list for processing
        pages = list(page_batch)

        # Extract images from pages
        images = []
        for page in pages:
            if hasattr(page, "image") and page.image:
                images.append(page.image)
            else:
                images.append(None)

        # Filter out None images for vLLM processing
        valid_images = [img for img in images if img is not None]

        if not valid_images:
            _log.warning("No valid images found in page batch")
            yield from pages
            return

        # Prepare prompts for vLLM
        prompt = """You are an OCR assistant. Analyze the provided image and:
1. Extract all text content exactly as it appears
2. Preserve formatting, structure, and layout
3. Include any mathematical formulas or special characters
4. Describe any tables, diagrams, or non-text elements

Return only the content without explanations."""

        prompts = []
        for img in valid_images:
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": img}})

        # Run vLLM batch inference
        try:
            outputs = self.llm.generate(prompts, self.sampling_params)

            # Extract generated text
            vlm_results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                vlm_results.append(generated_text)

            _log.info(f"✅ vLLM processed {len(vlm_results)} images")

            # Apply results back to pages
            result_idx = 0
            for i, page in enumerate(pages):
                if images[i] is not None:
                    # Create mock VLM response structure to match standard pipeline
                    if not hasattr(page, "predictions"):
                        page.predictions = type("obj", (object,), {})()
                    if not hasattr(page.predictions, "vlm_response"):
                        page.predictions.vlm_response = type("obj", (object,), {})()

                    page.predictions.vlm_response.text = vlm_results[result_idx]
                    result_idx += 1

                yield page

        except Exception as e:
            _log.error(f"❌ vLLM processing failed: {e}")
            # Fallback: yield pages without VLM processing
            yield from pages


class VllmBatchVlmPipeline(PaginatedPipeline):
    """vLLM-based VLM pipeline for batched inference with SmolDocling model."""

    def __init__(self, pipeline_options: VlmPipelineOptions):
        """Initialize the vLLM VLM pipeline."""
        super().__init__(pipeline_options)
        self.pipeline_options = pipeline_options
        self.keep_backend = True

        # Set up artifacts path
        artifacts_path: Optional[Path] = None
        if pipeline_options.artifacts_path is not None:
            artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

        # Configure build pipe with vLLM model
        _log.info("🚀 Initializing pipeline for VllmBatchVlmPipeline")

        self.build_pipe = [
            VllmBatchVlmModel(
                enabled=True,
                artifacts_path=artifacts_path,
                accelerator_options=pipeline_options.accelerator_options,
                vlm_options=None,  # We handle VLM options internally
            ),
        ]

        # Set up enrichment pipe (empty for now)
        self.enrichment_pipe = []

        # Image settings
        self.keep_images = self.pipeline_options.generate_page_images

        # Fallback pipeline for compatibility
        self.fallback_pipeline = VlmPipeline(pipeline_options)

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        """Initialize page resources."""
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()
        return page

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble the final document using VLM pipeline logic."""
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            # Use the standard VLM pipeline's assembly logic
            return self.fallback_pipeline._assemble_document(conv_res)

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        """Get default pipeline options."""
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend):
        """Check if backend is supported."""
        from docling.backend.pdf_backend import PdfDocumentBackend

        return isinstance(backend, PdfDocumentBackend)
