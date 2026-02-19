FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for audio I/O and inference.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Docker-specific runtime dependencies.
COPY docker/requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Copy project code.
COPY src/ src/
COPY infer.py infer.py
COPY config.yaml config.yaml

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
