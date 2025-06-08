#!/bin/bash

# Upload Docling models to GCS for caching
# Run this script once to populate the GCS bucket with models

set -e

# Configuration
GCS_BUCKET="jarvis-444507-docling-models"
MODELS_LIST="layout tableformer picture_classifier easyocr smoldocling"
LOCAL_MODELS_PATH="/tmp/docling-models"
DOCLING_SERVE_ARTIFACTS_PATH="${LOCAL_MODELS_PATH}/.cache/docling/models"

echo "=== Uploading Docling Models to GCS ==="
echo "Bucket: gs://${GCS_BUCKET}/"
echo "Models: ${MODELS_LIST}"
echo "========================================"

# Create local directory
mkdir -p "${LOCAL_MODELS_PATH}"
cd "${LOCAL_MODELS_PATH}"

# Download models locally first
echo "Downloading models locally..."
HF_HUB_DOWNLOAD_TIMEOUT="90" \
HF_HUB_ETAG_TIMEOUT="90" \
python -m docling.cli.models -o "${DOCLING_SERVE_ARTIFACTS_PATH}" ${MODELS_LIST}

# Upload to GCS
echo "Uploading models to GCS..."
gsutil -m cp -r "${DOCLING_SERVE_ARTIFACTS_PATH}/" "gs://${GCS_BUCKET}/models/"

echo "Upload complete!"
echo "Models are now available at: gs://${GCS_BUCKET}/models/"

# Cleanup
echo "Cleaning up local files..."
rm -rf "${LOCAL_MODELS_PATH}"

echo "=== GCS Model Upload Complete ===" 