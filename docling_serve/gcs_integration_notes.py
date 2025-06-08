"""
Integration notes for GCS model caching in run_job.py

To enable GCS model caching, add this to the top of run_conversion():

    # Enable GCS model caching (if using Dockerfile.gcs)
    if os.environ.get('USE_GCS_MODELS', '').lower() == 'true':
        from docling_serve.gcs_model_cache import setup_gcs_models
        setup_gcs_models()
        _log.info("GCS model caching initialized")

This should be added right before the VlmPipeline initialization in run_conversion().

Environment variables to set in Dockerfile.gcs:
- USE_GCS_MODELS=true
- GCS_BUCKET_NAME=jarvis-444507-docling-models (optional, has default)
"""
