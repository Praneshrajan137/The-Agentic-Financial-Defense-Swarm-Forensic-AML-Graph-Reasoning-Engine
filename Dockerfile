# Green Financial Crime Agent
# The Panopticon Protocol: Zero-Failure Synthetic Financial Crime Simulator
#
# Build: docker build -t green-financial-crime-agent .
# Run (Standalone): docker run -p 8000:8000 green-financial-crime-agent
# Run (Sidecar):    docker run -p 8000:5000 green-financial-crime-agent
#
# NOTE: TCMalloc (google-perftools) is used for high-performance memory management
# per The Panopticon Protocol Section 4.2.1

FROM python:3.10-slim

# ============================================================================
# STAGE 1: HIGH-PERFORMANCE MEMORY MANAGEMENT (TCMalloc)
# ============================================================================

# Install TCMalloc (google-perftools) for thread-caching memory allocation
# This prevents lock contention in multi-threaded graph operations
# Also install curl for health checks (more reliable than Python's requests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libgoogle-perftools4 \
    google-perftools \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Inject TCMalloc via LD_PRELOAD
# This must be set BEFORE any Python code runs
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# Verify TCMalloc is loaded (for debugging)
RUN echo "TCMalloc library path: $LD_PRELOAD" && \
    ls -la /usr/lib/x86_64-linux-gnu/libtcmalloc* && \
    echo "✅ TCMalloc successfully installed"

# ============================================================================
# STAGE 2: PYTHON DEPENDENCIES
# ============================================================================

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STAGE 3: APPLICATION CODE
# ============================================================================

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /mnt/user-data/outputs /app/outputs /app/data

# ============================================================================
# STAGE 4: HEALTH & RUNTIME CONFIGURATION
# ============================================================================

# Expose both ports for flexibility:
# - 5000: Internal port for sidecar proxy architecture
# - 8000: External port for standalone mode
EXPOSE 5000 8000

# Health check endpoint using curl (more reliable than Python's requests with TCMalloc)
# Uses ${PORT:-5000} to adapt to the port the server is actually listening on
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Environment variables
ENV GRAPH_SIZE=1000
ENV DIFFICULTY=5
ENV GENERATE_EVIDENCE=true
# Default port for health check (can be overridden)
ENV PORT=5000

# ============================================================================
# STAGE 5: STARTUP COMMAND
# ============================================================================

# Default command: Start FastAPI on 0.0.0.0:5000
# This allows both standalone and sidecar proxy architectures:
# - Standalone: docker run -p 8000:5000 ... OR override with --host 0.0.0.0 --port 8000
# - Sidecar: Envoy proxies external :8000 to internal :5000
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "5000"]
