FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Copy project
COPY src/ src/
COPY config.yaml .
COPY checkpoints/final/ checkpoints/final/

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
