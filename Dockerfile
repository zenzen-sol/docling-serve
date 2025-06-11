# GCS-optimized build - models loaded from GCS at runtime
# This dramatically reduces build time by skipping model downloads

# Stage 1: Base system setup (rarely changes)
FROM quay.io/sclorg/python-312-c9s:c9s AS base

USER 0

# Install system packages + Google Cloud SDK for gsutil
RUN --mount=type=bind,source=os-packages.txt,target=/tmp/os-packages.txt \
    dnf -y install --best --nodocs --setopt=install_weak_deps=False dnf-plugins-core && \
    dnf config-manager --best --nodocs --setopt=install_weak_deps=False --save && \
    dnf config-manager --enable crb && \
    dnf -y update && \
    # Add NVIDIA CUDA repository for CentOS Stream 9
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf -y install $(cat /tmp/os-packages.txt) && \
    # Install Google Cloud SDK for gsutil
    curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | tar -xzC /opt && \
    ln -s /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil && \
    dnf -y clean all && \
    rm -rf /var/cache/dnf

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

USER 1001
WORKDIR /opt/app-root/src

# Set environment variables (models will be set at runtime by GCS cache)
ENV \
    OMP_NUM_THREADS=4 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    TESSDATA_PREFIX=/usr/share/tesseract/tessdata/ \
    USE_GCS_MODELS=true

# Stage 2: Python dependencies (changes when dependencies change)
FROM base AS dependencies

# Copy dependency files only
COPY --chown=1001:0 pyproject.toml uv.lock ./

# Install dependencies in two stages (flash-attn requires special handling)
RUN uv venv /opt/app-root/venv && \
    . /opt/app-root/venv/bin/activate && \
    UV_HTTP_TIMEOUT=300 uv pip install .[cu128,tesserocr,rapidocr] && \
    echo "Building flash-attention from source with CUDA support..." && \
    export MAX_JOBS=4 && \
    export NVCC_THREADS=4 && \
    export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE && \
    UV_HTTP_TIMEOUT=1800 uv pip install flash-attn --no-build-isolation --verbose

ENV PATH="/opt/app-root/venv/bin:$PATH"

# Fix permissions for venv
RUN chown -R 1001:0 /opt/app-root/venv && \
    chmod -R g=u /opt/app-root/venv

# Stage 3: Final stage with project code (NO model downloads!)
FROM dependencies AS final

# Copy project files
COPY --chown=1001:0 ./docling_serve ./docling_serve

# Create temp directory for model caching and gcloud config as root
USER root
RUN mkdir -p /tmp/docling-models && \
    chown -R 1001:0 /tmp/docling-models && \
    chmod -R g=u /tmp/docling-models && \
    mkdir -p /opt/app-root/src/.config/gcloud/configurations && \
    chown -R 1001:0 /opt/app-root/src/.config && \
    chmod -R g=u /opt/app-root/src/.config

# Switch back to user 1001
USER 1001

ENTRYPOINT ["python", "-m", "docling_serve.run_job"] 