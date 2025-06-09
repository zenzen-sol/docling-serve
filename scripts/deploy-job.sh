#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Source environment variables if .env file exists
if [ -f .env ]; then
  source .env
fi

# Check for required environment variables
required_vars=( "GCP_PROJECT_ID" "ARTIFACT_REGISTRY_REGION" "IMAGE_REPO_NAME" "IMAGE_NAME" "IMAGE_TAG" "CLOUD_RUN_JOB_NAME" "CLOUD_RUN_REGION" )
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: Environment variable ${var} is not set. Please create a .env file or export the variable." >&2
    exit 1
  fi
done

# Construct the full image path
FULL_IMAGE_PATH="${ARTIFACT_REGISTRY_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${IMAGE_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Pushing image to Google Artifact Registry: ${FULL_IMAGE_PATH}"

# Authenticate with Google Cloud and push the image
gcloud auth configure-docker "${ARTIFACT_REGISTRY_REGION}-docker.pkg.dev"
docker push "${FULL_IMAGE_PATH}"

echo "Image pushed successfully."
echo "Deploying image to Cloud Run job: ${CLOUD_RUN_JOB_NAME} in region: ${CLOUD_RUN_REGION}"

gcloud beta run jobs deploy "${CLOUD_RUN_JOB_NAME}" \
    --image "${FULL_IMAGE_PATH}" \
    --region "${CLOUD_RUN_REGION}" \
    --cpu 4 \
    --memory 16Gi \
    --task-timeout 3600 \
    --max-retries 1 \
    --set-env-vars DOCLING_SERVE_MAX_SYNC_WAIT=300,USE_GCS_MODELS=true \
    --gpu 1 \
    --gpu-type nvidia-l4

echo "Deployment command executed."
echo "You can check the status in the Google Cloud Console or by running: gcloud run jobs describe ${CLOUD_RUN_JOB_NAME} --region ${CLOUD_RUN_REGION}"