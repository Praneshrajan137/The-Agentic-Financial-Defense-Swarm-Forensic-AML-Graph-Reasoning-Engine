# ═══════════════════════════════════════════════════════════════════
# Purple Agent -- Project Gamma | Dockerfile v12.0
# The Panopticon Protocol: Zero-Failure Financial Crime Detection
#
# Build: docker build -t purple-agent .
# Run:   docker run -p 8080:8080 purple-agent
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim-bookworm

# Determinism: fixed hash seed for dict/set iteration order
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# TCMalloc for high-performance memory management
# x86_64 only (primary target). ARM64: /usr/lib/aarch64-linux-gnu/...
RUN apt-get update && \
    apt-get install -y --no-install-recommends libtcmalloc-minimal4 && \
    rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

WORKDIR /app

# Layer-cached dependency install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model at build time (66MB)
RUN python -m spacy download en_core_web_sm

# Copy application code (after deps for cache)
COPY . .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash purple && \
    chown -R purple:purple /app
USER purple

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
