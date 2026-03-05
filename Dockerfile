# Green Financial Crime Agent v7.0
# The Panopticon Protocol: Zero-Failure Synthetic Financial Crime Simulator
#
# Build: docker build -t green-financial-crime-agent .
# Run:   docker run -p 9090:9090 green-financial-crime-agent
#
# Purple Agent connects to this server at http://localhost:9090/a2a

FROM python:3.11-slim

# ============================================================================
# STAGE 1: SYSTEM DEPENDENCIES + TCMalloc
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libgoogle-perftools4 \
    google-perftools \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# TCMalloc: configure LD_PRELOAD so the allocator is actually used at runtime.
# We prefer the canonical x86_64 path but fall back to auto-detection on other archs.
RUN TCMALLOC_PATH_DEFAULT="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" && \
    if [ -f "$TCMALLOC_PATH_DEFAULT" ]; then \
        echo "Using TCMalloc at $TCMALLOC_PATH_DEFAULT"; \
        echo "LD_PRELOAD=$TCMALLOC_PATH_DEFAULT" >> /etc/environment; \
    else \
        TCMALLOC_PATH_DETECTED=$(find /usr/lib -name "libtcmalloc_minimal.so*" -type f | head -1 || true); \
        if [ -n "$TCMALLOC_PATH_DETECTED" ]; then \
            echo "Using detected TCMalloc at $TCMALLOC_PATH_DETECTED"; \
            echo "LD_PRELOAD=$TCMALLOC_PATH_DETECTED" >> /etc/environment; \
        else \
            echo "WARNING: TCMalloc not found, proceeding without it"; \
        fi; \
    fi

# Export LD_PRELOAD into the container runtime environment if the library exists.
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# ============================================================================
# STAGE 2: PYTHON DEPENDENCIES
# ============================================================================

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STAGE 3: APPLICATION CODE
# ============================================================================

COPY . .
RUN mkdir -p /mnt/user-data/outputs /app/outputs /app/data

# ============================================================================
# STAGE 4: HEALTH & RUNTIME CONFIGURATION
# ============================================================================

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

ENV GRAPH_SIZE=1000 \
    DIFFICULTY=5 \
    GENERATE_EVIDENCE=true \
    A2A_SERVER_PORT=9090

# ============================================================================
# STAGE 5: STARTUP
# ============================================================================

CMD ["python", "main.py", "serve", "--generate-on-startup", "--host", "0.0.0.0", "--port", "9090"]
