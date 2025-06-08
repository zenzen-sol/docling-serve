#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Source environment variables if .env file exists
if [ -f .env ]; then
  source .env
fi

# Check for required environment variables
required_vars=( "GCP_PROJECT_ID" "ARTIFACT_REGISTRY_REGION" "IMAGE_REPO_NAME" "IMAGE_NAME" "IMAGE_TAG" )
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: Environment variable ${var} is not set. Please create a .env file or export the variable." >&2
    exit 1
  fi
done

# Construct the full image path
FULL_IMAGE_PATH="${ARTIFACT_REGISTRY_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${IMAGE_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building job image for linux/amd64: ${FULL_IMAGE_PATH}"

# Build the container image for the correct production architecture, pointing to the correct Dockerfile and context
docker build --platform linux/amd64 -t "${FULL_IMAGE_PATH}" -f ../Dockerfile ..

echo "Build complete."
echo "To push the image, run: gcloud auth configure-docker ${ARTIFACT_REGISTRY_REGION}-docker.pkg.dev && docker push ${FULL_IMAGE_PATH}"
echo "To deploy the job, run: ./deploy-job.sh"
echo "Cleaning up Docker build cache to save space..."

# Forcefully prune the build cache to remove old layers
docker builder prune --force

echo "Cleanup complete." 