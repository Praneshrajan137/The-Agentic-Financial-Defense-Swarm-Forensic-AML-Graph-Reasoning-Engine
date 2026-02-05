# Green Agent Dockerfile for Scenarios
# The Panopticon Protocol - Sidecar Architecture
#
# This is used by scenarios/financial_crime/docker-compose.yml
# Optimized for Envoy sidecar pattern with TCMalloc

FROM python:3.10-slim

# ============================================================================
# STAGE 1: HIGH-PERFORMANCE MEMORY MANAGEMENT (TCMalloc)
# ============================================================================

# Install TCMalloc and curl for reliable health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libgoogle-perftools4 \
    google-perftools \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Inject TCMalloc via LD_PRELOAD
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# Verify TCMalloc
RUN echo "TCMalloc library path: $LD_PRELOAD" && \
    ls -la /usr/lib/x86_64-linux-gnu/libtcmalloc* && \
    echo "✅ TCMalloc successfully installed"

# ============================================================================
# STAGE 2: PYTHON DEPENDENCIES
# ============================================================================

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements from root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STAGE 3: APPLICATION CODE
# ============================================================================

COPY . .
RUN mkdir -p /app/outputs /app/data

# ============================================================================
# STAGE 4: HEALTH & RUNTIME
# ============================================================================

EXPOSE 5000

# Health check using curl (reliable with TCMalloc, unlike Python's requests)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:5000/health || curl -f http://127.0.0.1:5000/health || exit 1

ENV GRAPH_SIZE=1000
ENV DIFFICULTY=5
ENV GENERATE_EVIDENCE=true
ENV PORT=5000

# ============================================================================
# STAGE 5: STARTUP - Binds to 0.0.0.0 for container networking
# ============================================================================

# CRITICAL: Must bind to 0.0.0.0 for Docker container networking to work
# 127.0.0.1 only allows connections from INSIDE the container
# Envoy sidecar connects from its own container, needs 0.0.0.0
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "5000"]
