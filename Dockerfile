# Use official Python 3.11 slim image
FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    antiword \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Copy startup script
COPY start.sh /start.sh

# Make startup script executable
RUN chmod +x /start.sh

# Create non-root user
RUN useradd -m appuser

# Switch to non-root user (recommended)
USER appuser

# Start application
CMD ["/start.sh"]