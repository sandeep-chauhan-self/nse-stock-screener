# Multi-stage Dockerfile for NSE Stock Screener
# Stage 1: Build environment with dependencies
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Production runtime
FROM python:3.11-slim AS runtime

# Install runtime system dependencies and create non-root user for security
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash screener \
    && mkdir -p /app \
    && chown -R screener:screener /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=screener:screener . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/output /app/output/reports /app/output/charts /app/output/backtests && \
    chown -R screener:screener /app

# Switch to non-root user
USER screener

# Set environment variables for the application
ENV PYTHONPATH="/app/src:$PYTHONPATH" \
    NSE_SCREENER_CONFIG_PATH="/app/config" \
    NSE_SCREENER_DATA_PATH="/app/data" \
    NSE_SCREENER_OUTPUT_PATH="/app/output"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from scripts.check_deps import main; main()" || exit 1

# Expose port for potential web interface
EXPOSE 8000

# Default command - run dependency check and then interactive mode
CMD ["python", "-m", "src.enhanced_early_warning_system", "--help"]

# Alternative commands:
# For analysis: docker run nse-screener python -m src.enhanced_early_warning_system
# For backtesting: docker run nse-screener python -m src.advanced_backtester
# For shell access: docker run -it nse-screener /bin/bash

# Build instructions:
# docker build -t nse-screener:latest .
# docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output nse-screener:latest