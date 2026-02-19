#!/bin/bash

# Build and run TechnicalTTS server in Docker

set -e

echo "Building Docker image..."
docker build -f docker/Dockerfile -t technical-tts .

echo ""
echo "Starting server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

docker run --rm -p 8000:8000 --name technical-tts technical-tts
