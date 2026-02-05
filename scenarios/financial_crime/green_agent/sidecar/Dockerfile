# scenarios/financial_crime/green_agent/sidecar/Dockerfile

FROM rust:1.75-slim as builder

WORKDIR /build

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy actual source
COPY src ./src

# Build the sidecar
RUN cargo build --release

# ============================================================================
# RUNTIME IMAGE
# ============================================================================
FROM debian:bookworm-slim

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/green-agent-sidecar /app/

# Expose external port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the sidecar
CMD ["/app/green-agent-sidecar"]
