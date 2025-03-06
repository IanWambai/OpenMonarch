# Use official PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install required Python libraries
RUN pip install --no-cache-dir torch transformers fastapi uvicorn runpod

# Copy model and API script
COPY . /app

# Expose API port
EXPOSE 8000

# Start API on container launch
CMD ["uvicorn", "monarch_api:app", "--host", "0.0.0.0", "--port", "8000"]
