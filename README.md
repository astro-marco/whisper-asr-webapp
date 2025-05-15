# `whisper-asr-webapp`

[![Docker](https://github.com/FluxCapacitor2/whisper-asr-webapp/actions/workflows/docker.yml/badge.svg)](https://github.com/FluxCapacitor2/whisper-asr-webapp/actions/workflows/docker.yml)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/FluxCapacitor2/whisper-asr-webapp/main)

A web app for automatic speech recognition using OpenAI's Whisper model running locally.

```sh
# Quickstart with Docker:
docker run --rm -it -p 8000:8000 -v whisper_models:/root/.cache/whisper ghcr.io/fluxcapacitor2/whisper-asr-webapp:main
```

![](/.github/readme_images/app_dark.png#gh-dark-mode-only)
![](/.github/readme_images/app_light.png#gh-light-mode-only)

## Features

- Customize the model, language, and initial prompt
- Enable per-word timestamps (visible in downloaded JSON output)
- Speaker diarization to distinguish different voices in the transcript
- Runs Whisper locally
- Pre-packaged into a single Docker image
- View timestamped transcripts in the app
- Download transcripts in plain text, VTT, SRT, TSV, or JSON formats

## Architecture

The frontend is built with Svelte and builds to static HTML, CSS, and JS.

The backend is built with FastAPI. The main endpoint, `/transcribe`, pipes an uploaded file into ffmpeg, then into Whisper. For speaker diarization, WhisperX is also employed to assign speaker labels. Once transcription is complete, it's returned as a JSON payload.

In a containerized environment, the static assets from the frontend build are served by the same FastAPI (Uvicorn) server that handles transcription.

## Running

The primary way to run the application is using Docker.

1. Ensure Docker is running.
2. Use the provided `run.sh` script from the project root:
   ```sh
   ./run.sh
   ```
   This script builds the Docker image (if it doesn't exist or if changes are detected) and starts the container. It also mounts a volume (`whisper_models`) to cache downloaded Whisper models.
3. Visit http://localhost:8000 in a web browser.

If you prefer more direct control, you can still use the standard Docker commands:

- For an interactive terminal: `docker run --rm -it -p 8000:8000 -v whisper_models:/root/.cache/whisper ghcr.io/fluxcapacitor2/whisper-asr-webapp:main` (or your locally built image name like `fluxcapacitor2/whisper-asr-webapp:local-dev`).
- To run in the background: `docker run -d -p 8000:8000 -v whisper_models:/root/.cache/whisper ghcr.io/fluxcapacitor2/whisper-asr-webapp:main`.

## Development

For development, you can use the provided shell scripts to streamline your workflow.

The `run.sh` script offers a Docker-based approach. It builds the image (if needed, incorporating your local code changes) and runs the container, closely mirroring the production setup:

```sh
./run.sh
```

If you prefer a faster local iteration cycle, especially for backend changes, `run-local.sh` is helpful. It builds the frontend and starts the backend server using your local Python environment (ensure you've set up `backend/.venv` with `poetry install` first):

```sh
./run-local.sh
```

With this setup, backend Python changes will often auto-reload. For frontend-only modifications, `update-ui.sh` quickly rebuilds the UI assets:

```sh
./update-ui.sh
```

Whichever script you use to start it, the app will be available at http://localhost:8000.
Note: When using `run.sh`, code changes require stopping the script and re-running it to rebuild the Docker image.
With `run-local.sh`:

- Backend Python changes typically auto-reload the server (due to Uvicorn's `--reload` flag).
- Frontend changes require rebuilding the UI assets. You can achieve this by:
  - Re-running `run-local.sh` (it rebuilds the UI before starting the server).
  - Running `update-ui.sh` separately (if the server is already running, you'll then need to refresh your browser to see the UI changes).
