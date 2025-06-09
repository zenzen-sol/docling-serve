# Docling-Serve Cloud Run Jobs Guide

## 🎯 Quick Start

This guide shows you how to build and deploy `docling-serve` as a Google Cloud Run Job for processing documents with GPU acceleration and flash-attention.

### Prerequisites

- Google Cloud SDK installed and authenticated
- Docker installed
- Access to the `jarvis-444507` Google Cloud project

### Build and Deploy (5 minutes)

```bash
cd docling-serve/scripts
./build.sh && ./deploy-job.sh
```

That's it! The job is now deployed and ready to process documents.

## 🏗️ How It Works

### Current Architecture

- **Docker Image**: Multi-stage build with GCS model caching
- **Models**: Pre-downloaded from GCS at runtime (fast cold starts)
- **GPU**: NVIDIA T4 with flash-attention optimization
- **Processing**: PDF → Docling JSON with VLM pipeline

### Build Process

1. **Base Stage**: System packages + Google Cloud SDK
2. **Dependencies**: Python packages + flash-attention
3. **Runtime**: Copy code, download models from GCS
4. **Deploy**: Push to Cloud Run Jobs

### Why GCS Model Caching?

- ✅ **Fast builds**: 5-15 minutes (vs 45+ minutes downloading models)
- ✅ **Fast cold starts**: 60-90 seconds (models download from GCS)
- ✅ **Reliability**: No build failures from model download timeouts

## 📋 Step-by-Step Guide

### 1. Build the Docker Image

```bash
cd docling-serve/scripts
./build.sh
```

**What this does**:
- Builds multi-stage Docker image with flash-attention support
- Configures GCS model caching
- Pushes to Google Container Registry
- Takes ~5-15 minutes

### 2. Deploy to Cloud Run Jobs

```bash
./deploy-job.sh
```

**What this does**:
- Creates/updates Cloud Run Job with GPU (NVIDIA T4)
- Sets memory to 16GB, CPU to 4 vCPUs
- Configures timeout to 60 minutes
- Takes ~2-3 minutes

### 3. Test the Deployment

The job is triggered by your maboroshi API when documents need processing. Check logs in:
- **Google Cloud Console** → Cloud Run Jobs → docling-serve-job → Logs

## 🔧 Configuration

### Flash-Attention Settings

The job is configured with flash-attention for faster processing:

```python
# In run_job.py
pipeline_options = VlmPipelineOptions(
    accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=True)
)
# Disable 8-bit quantization (conflicts with flash-attention)
custom_vlm_options = smoldocling_vlm_conversion_options.model_copy(
    update={"load_in_8bit": False}
)
```

### Resource Limits

- **GPU**: 1x NVIDIA T4
- **Memory**: 16GB
- **CPU**: 4 vCPUs  
- **Timeout**: 60 minutes per job


## 📁 Key Files

```
docling-serve/
├── Dockerfile                    # Multi-stage build with GCS caching
├── scripts/
│   ├── build.sh                  # Build Docker image
│   ├── deploy-job.sh             # Deploy to Cloud Run Jobs
│   └── upload-models-to-gcs.sh   # One-time model upload (already done)
├── docling_serve/
│   ├── run_job.py                # Main job execution with flash-attention
│   ├── gcs_model_cache.py        # GCS model downloading
│   └── http_logging.py           # Progress reporting
└── CLOUD_JOBS_README.md          # This guide
```

## 🚀 Development Workflow

### Code Changes Only (Fast: ~5 minutes)
```bash
# 1. Make your code changes
# 2. Build and deploy
cd scripts && ./build.sh && ./deploy-job.sh
```

### Dependency Changes (Moderate: ~15-20 minutes)
Same as above - the multi-stage build handles this efficiently.

### Model Changes (Slow: ~45+ minutes)
If you need different models, update `MODELS_LIST` in `Dockerfile` and rebuild.

## 🎯 Performance Expectations

- **Cold Start**: 60-90 seconds (downloading models from GCS)
- **Processing**: ~30-60 seconds per page with flash-attention
- **Memory Usage**: ~8-12GB for typical documents
- **GPU Utilization**: High during VLM processing phases

## 🔍 Monitoring

### Check Job Status
```bash
gcloud run jobs describe docling-serve-job --region=us-central1
```

### View Recent Executions
```bash
gcloud run jobs executions list --job=docling-serve-job --region=us-central1
```

### Stream Logs
```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=docling-serve-job" --limit=50 --format="table(timestamp,textPayload)"
```

---

**Need help?** Check the logs first, then verify your build and deployment steps above. 