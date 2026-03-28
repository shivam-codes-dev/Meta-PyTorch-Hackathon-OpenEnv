FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch numpy

# Copy all project files
COPY . .

# Command to run the inference script
CMD ["python", "inference.py"]
