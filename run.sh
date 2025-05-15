#!/usr/bin/env bash

# Check that Docker daemon is running
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running. Please start Docker Desktop (macOS) or your Docker service."
  exit 1
fi

# Build (with ARM64 support if needed)
docker buildx build \
  --platform linux/arm64 \
  --load \
  -t fluxcapacitor2/whisper-asr-webapp:local-dev .

# Run, mounting whisper_models to cache downloaded model files
docker run \
  -p 8000:8000 \
  -v whisper_models:/root/.cache/whisper \
  --rm -it fluxcapacitor2/whisper-asr-webapp:local-dev
