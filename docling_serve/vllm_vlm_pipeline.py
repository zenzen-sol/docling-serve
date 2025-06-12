"""
vLLM-based VLM Pipeline for batched inference.

Surgical replacement of VLM backend only - keeps all Docling functionality
while enabling GPU batching for massive performance gains.
"""

import gc
import logging
import os
import time
from typing import Iterable

import torch

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.settings import settings
from docling.pipeline.vlm_pipeline import VlmPipeline

_log = logging.getLogger(__name__)


class VllmBatchVlmModel:
    """Optimized vLLM-based batch VLM model with aggressive diagnostics and engine management."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self._batch_counter = 0
        self._total_processing_time = 0
        self._total_pages_processed = 0

        # Aggressive engine management - restart every batch for now
        self._engine_restart_threshold = 5

        self._init_engine()

    def _init_engine(self):
        """Initialize or reinitialize the vLLM engine."""
        try:
            self.logger.info("🔄 Initializing vLLM engine...")
            start_time = time.time()

            # Use SmolDocling - specifically designed for document processing
            model_id = "ds4sd/SmolDocling-256M-preview"

            # Import vLLM dynamically to handle missing dependencies
            try:
                from vllm import LLM, SamplingParams
            except ImportError as import_error:
                self.logger.error(f"❌ vLLM not available: {import_error}")
                self.logger.warning("⚡ Falling back to standard VLM pipeline")
                self.llm = None
                self.sampling_params = None
                return

            # Aggressive cleanup before initialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # Conservative vLLM parameters to prevent degradation
            vllm_params = {
                "model": model_id,
                "gpu_memory_utilization": float(
                    os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75")
                ),
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "enforce_eager": True,
                "max_num_seqs": 4,  # Conservative: match batch size exactly
                "max_model_len": 2048,  # CRITICAL: Restored to handle complex pages
                "disable_log_stats": True,  # Reduce logging overhead
                # Additional stability parameters
                "enable_prefix_caching": False,  # Disable caching to prevent memory issues
                "swap_space": 0,  # Disable swap to prevent memory fragmentation
                "max_num_batched_tokens": 8192,  # Conservative token limit
            }

            self.llm = LLM(**vllm_params)

            self.sampling_params = SamplingParams(
                max_tokens=6000,  # Restored for full page content
                temperature=0.0,
                top_p=0.95,
                stop=["<end_of_utterance>"],
            )

            init_time = time.time() - start_time
            self.logger.info(
                f"✅ vLLM engine initialized in {init_time:.2f}s (max_model_len: 2048)"
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vLLM engine: {e}")
            self.llm = None
            self.sampling_params = None

    def _get_engine_diagnostics(self):
        """Get detailed vLLM engine diagnostics."""
        if not self.llm or not hasattr(self.llm, "llm_engine"):
            return {"status": "no_engine"}

        try:
            engine = self.llm.llm_engine
            scheduler = getattr(engine, "scheduler", None)

            diagnostics = {
                "scheduler_exists": scheduler is not None,
                "waiting_requests": len(getattr(scheduler, "waiting", []))
                if scheduler
                else 0,
                "running_requests": len(getattr(scheduler, "running", []))
                if scheduler
                else 0,
                "swapped_requests": len(getattr(scheduler, "swapped", []))
                if scheduler
                else 0,
            }

            # Try to get cache info
            if hasattr(engine, "cache_config"):
                cache_config = engine.cache_config
                diagnostics["cache_blocks"] = getattr(
                    cache_config, "num_gpu_blocks", "unknown"
                )

            # Try to get model runner info
            if hasattr(engine, "model_executor"):
                model_executor = engine.model_executor
                diagnostics["model_executor_exists"] = model_executor is not None

            return diagnostics

        except Exception as e:
            return {"error": str(e)}

    def _force_cleanup(self):
        """Aggressive cleanup of GPU memory and Python objects."""
        try:
            # Multiple rounds of Python garbage collection
            for _ in range(3):
                gc.collect()

            # CUDA cleanup
            if torch.cuda.is_available():
                # Clear all cached memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Reset memory stats
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except:
                    pass

                # Force garbage collection again after CUDA cleanup
                gc.collect()

            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def _restart_engine(self):
        """Restart the vLLM engine to prevent degradation."""
        try:
            self.logger.info("Restarting vLLM engine to prevent degradation...")

            # Cleanup before restart
            self._force_cleanup()

            if hasattr(self, "llm") and self.llm is not None:
                try:
                    # Try to properly shutdown the engine
                    if hasattr(self.llm, "llm_engine"):
                        self.llm.llm_engine.stop_remote_worker_execution_loop()
                except:
                    pass
                del self.llm

            # Aggressive cleanup after deletion
            self._force_cleanup()

            # Small delay to ensure cleanup
            time.sleep(2)

            # Reinitialize
            self._init_engine()

            self.logger.info("vLLM engine restarted successfully")
        except Exception as e:
            self.logger.error(f"Failed to restart vLLM engine: {e}")
            raise

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]):
        """Process page batch with comprehensive diagnostics and progress tracking."""
        pages = list(page_batch)

        if not self.llm or not pages:
            yield from pages
            return

        # Collect pages with images for batch processing
        batch_pages = []
        batch_images = []

        for page in pages:
            if hasattr(page, "image") and page.image is not None:
                batch_pages.append(page)
                batch_images.append(page.image)

        if not batch_images:
            yield from pages
            return

        self._batch_counter += 1
        batch_start_time = time.time()

        # Engine restart strategy - restart every 5 batches to prevent degradation
        if self._batch_counter % self._engine_restart_threshold == 0:
            _log.info(f"🔄 Restarting vLLM engine (batch #{self._batch_counter})...")

            # Destroy current engine
            if self.llm:
                try:
                    del self.llm
                except:
                    pass
                self.llm = None

            # Aggressive cleanup
            self._force_cleanup()

            # Reinitialize
            self._init_engine()

            if not self.llm:
                _log.error("❌ Engine restart failed")
                yield from pages
                return

        _log.info(
            f"🔄 Processing batch #{self._batch_counter} ({len(batch_images)} pages)"
        )

        # Pre-batch diagnostics (reduced frequency)
        if self._batch_counter % 5 == 1:  # Log every 5th batch
            pre_diagnostics = self._get_engine_diagnostics()
            _log.info(f"📊 Engine state: {pre_diagnostics}")

        # Prepare prompts
        chat_template = "<|im_start|>User:<image>Convert this page to docling.<end_of_utterance>Assistant:"
        prompts = []
        for image in batch_images:
            prompts.append(
                {"prompt": chat_template, "multi_modal_data": {"image": image}}
            )

        # Run inference with detailed timing and degradation detection
        try:
            # Set up timeout
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Batch #{self._batch_counter} timed out after 5 minutes"
                )

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minutes

            try:
                inference_start = time.time()

                # Monitor for early signs of degradation
                _log.info(
                    f"🚀 Starting inference for batch #{self._batch_counter} at {time.strftime('%H:%M:%S')}"
                )

                outputs = self.llm.generate(prompts, self.sampling_params)
                inference_time = time.time() - inference_start

                # Check for degradation signs
                avg_time_per_prompt = inference_time / len(prompts)
                if (
                    avg_time_per_prompt > 15.0
                ):  # More than 15s per prompt indicates degradation
                    _log.warning(
                        f"⚠️ Slow processing detected: {avg_time_per_prompt:.1f}s/prompt (batch #{self._batch_counter})"
                    )
                    _log.warning(
                        "🔄 Consider reducing engine restart threshold if this persists"
                    )

                _log.info(
                    f"⚡ Batch #{self._batch_counter}: {inference_time:.2f}s inference ({len(outputs)} outputs)"
                )

            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # Process outputs with individual page progress logging
            for i, output in enumerate(outputs):
                page = batch_pages[i]
                generated_doctags = output.outputs[0].text

                if not hasattr(page, "predictions") or page.predictions is None:
                    page.predictions = PagePredictions()

                page.predictions.vlm_response = VlmPrediction(text=generated_doctags)

                # Log individual page completion for progress tracking
                _log.info(
                    f"Page {page.page_no}: DocTags stored ({len(generated_doctags)} chars)"
                )

                # Log sample for first few pages only
                if self._batch_counter <= 3:
                    sample = generated_doctags[:100].replace("\n", " ")
                    _log.info(f"Page {page.page_no}: DocTags sample - {sample}...")

            # Batch timing summary
            batch_time = time.time() - batch_start_time
            self._total_processing_time += batch_time
            self._total_pages_processed += len(batch_pages)

            avg_time_per_page = batch_time / len(batch_pages)
            overall_avg = (
                self._total_processing_time / self._total_pages_processed
                if self._total_pages_processed > 0
                else 0
            )

            _log.info(
                f"✅ Batch #{self._batch_counter}: {batch_time:.2f}s total, {avg_time_per_page:.2f}s/page (avg: {overall_avg:.2f}s/page)"
            )

            # Cleanup after each batch
            self._force_cleanup()

        except TimeoutError as e:
            _log.error(f"⏰ {e}")
            # Force engine restart on timeout
            self._batch_counter = self._engine_restart_threshold - 1

        except Exception as e:
            _log.error(f"❌ Batch #{self._batch_counter} failed: {e}")
            # Force engine restart on error
            self._batch_counter = self._engine_restart_threshold - 1

        yield from pages


class VllmBatchVlmPipeline(VlmPipeline):
    """
    VLM Pipeline with optimized vLLM batching and degradation prevention.

    Key optimizations:
    - Proper context length (max_model_len=2048) for complex document pages
    - Proactive engine restart (every 5 batches) to prevent vLLM degradation
    - Comprehensive error handling and 5-minute timeout protection
    - DocTags format preservation for bounding box information
    - Early degradation detection and performance monitoring

    Known Issue: vLLM vision-language models suffer from engine degradation
    after multiple batches, causing complete processing hangs. The restart
    strategy prevents this by refreshing the engine before degradation occurs.

    Performance: ~3.5-4.0 pages/minute with GPU acceleration
    """

    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        _log.info("🔧 Initialized VlmPipeline with aggressive diagnostics enabled")

        if self.build_pipe:
            original_model_name = type(self.build_pipe[0]).__name__
            self.build_pipe[0] = VllmBatchVlmModel()
            _log.info(
                f"✅ Replaced '{original_model_name}' with diagnostic VllmBatchVlmModel"
            )
            _log.info(
                f"🏷️ DocTags parser: {type(self.build_pipe[1]).__name__ if len(self.build_pipe) > 1 else 'None'}"
            )
        else:
            _log.warning("Build pipe is empty, cannot replace model.")
            self.build_pipe = [VllmBatchVlmModel()]

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend):
        from docling.backend.pdf_backend import PdfDocumentBackend

        return isinstance(backend, PdfDocumentBackend)
