# Use Python 3.9 slim as base image (smaller than full python image)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for torch, numpy, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port if needed (optional, remove if not using web server)
# EXPOSE 8000

# Default command: run inference.py
CMD ["python", "inference.py"]
