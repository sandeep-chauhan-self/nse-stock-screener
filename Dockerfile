# NSE Stock Screener - Production Dockerfile
# Multi-stage build with security hardening
# Based on official Python slim image with security updates

# ===== BUILD STAGE =====
FROM python:3.11-slim as builder

# Set build arguments for security
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF

# Install build dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create build directory
WORKDIR /build

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./
COPY src/security/requirements-security.txt ./

# Create virtual environment with security in mind
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-security.txt

# ===== PRODUCTION STAGE =====
FROM python:3.11-slim as production

# Metadata labels for security tracking
LABEL maintainer="NSE Stock Screener Team" \
      version="1.0.0" \
      description="Hardened NSE Stock Screener with security best practices" \
      security.non-root="true" \
      security.minimal-base="true" \
      security.updates="included" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Install only essential runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user with minimal privileges
RUN groupadd -r nse-screener --gid=1000 && \
    useradd -r -g nse-screener --uid=1000 \
    --home-dir=/app --shell=/bin/bash \
    --comment="NSE Screener Application User" \
    nse-screener

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set secure environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    NSE_SCREENER_ENV=production \
    NSE_SCREENER_LOG_LEVEL=INFO \
    NSE_SCREENER_DATA_PATH=/app/data \
    NSE_SCREENER_OUTPUT_PATH=/app/output \
    NSE_SCREENER_LOGS_PATH=/app/logs

# Create application directory structure with proper permissions
WORKDIR /app
RUN mkdir -p /app/data /app/output /app/logs /app/temp && \
    mkdir -p /app/output/reports /app/output/charts /app/output/backtests && \
    chown -R nse-screener:nse-screener /app

# Copy application code with proper ownership
COPY --chown=nse-screener:nse-screener src/ ./src/
COPY --chown=nse-screener:nse-screener data/ ./data/
COPY --chown=nse-screener:nse-screener scripts/ ./scripts/

# Copy Docker-specific files
COPY --chown=nse-screener:nse-screener docker/entrypoint.sh ./
COPY --chown=nse-screener:nse-screener docker/healthcheck.py ./

# Make scripts executable
RUN chmod +x entrypoint.sh healthcheck.py

# Security: Remove unnecessary packages and clean up
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

# Switch to non-root user for security
USER nse-screener

# Expose application port (non-privileged port)
EXPOSE 8000

# Add health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Use dumb-init as PID 1 to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Default command with proper signal handling
CMD ["./entrypoint.sh"]

# Security hardening: Run as non-root, minimal attack surface
# Build with: docker build --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') --build-arg VCS_REF=$(git rev-parse HEAD) -t nse-screener:latest .
# Run with: docker run --read-only --tmpfs /tmp --tmpfs /app/temp -v $(pwd)/data:/app/data:ro -v $(pwd)/output:/app/output nse-screener:latest