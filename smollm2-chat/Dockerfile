FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and cleanup in a single layer
RUN apt-get update --allow-insecure-repositories && \
    mkdir -p /var/cache/apt/archives/partial && \
    apt-get clean && \
    apt-get install -y --allow-unauthenticated --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install CPU-only PyTorch and other dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Command to run the application
CMD ["python", "app.py"]