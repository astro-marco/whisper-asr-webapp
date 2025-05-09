from dotenv import load_dotenv
load_dotenv()  # load variables from .env file
import os
from pathlib import Path
from typing import Annotated, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import ffmpeg
import whisper
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx import assign_word_speakers
import numpy as np
import requests
from datetime import timedelta

app = FastAPI()


app.add_middleware(CORSMiddleware, allow_origins=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
static_dir = BASE_DIR / "static"
if static_dir.exists():
    print(f"Serving UI from {static_dir}")
    # serve UI static files under /static
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

@app.get("/")
def root():
    # redirect root to the UI index.html
    return RedirectResponse(url="/static/index.html")


download_size_cache = {}


@app.get("/size")
def download_size(model: str):
    url = whisper._MODELS[model]
    if url in download_size_cache:
        return download_size_cache[url]
    res = requests.head(url)
    size = int(res.headers.get("Content-Length"))
    download_size_cache[url] = size
    return size


@app.get("/models")
def models():
    models = {}
    root = os.path.join(
        os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
        "whisper",
    )
    for model in whisper.available_models():
        models[model] = os.path.exists(f"{root}/{model}.pt")
        # Some models are aliases for others; e.g. "turbo" -> "large-v3-turbo".
        # If a model file with an aliased name is present, the model is already downloaded and should be marked as such.
        for other, url in whisper._MODELS.items():
            if url == whisper._MODELS[model] and os.path.exists(f"{root}/{other}.pt"):
                models[model] = True
    
    return models


@app.post("/transcribe")
def transcribe(
    file: Annotated[UploadFile, File()],
    task: Annotated[str, Form()] = "transcribe",
    model: Annotated[str, Form()] = "base",
    initial_prompt: Annotated[str, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = False,
    language: Annotated[str, Form()] = None,
    diarize: Annotated[bool, Form()] = False,
    num_speakers: Annotated[Optional[int], Form()] = None,
):
    bytes = file.file.read()
    np_array = convert_audio(bytes)
    whisper_instance = whisper.load_model(model)
    # sanitize language: treat empty or placeholder "string" as None
    lang = None if not language or language.lower() == "string" else language
    # call Whisper with sanitized language parameter
    result = whisper_instance.transcribe(
        audio=np_array,
        verbose=True,
        task=task,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
        language=lang,
    )
    # if diarization requested, run WhisperX pipeline
    # WhisperX diarization pipeline expects a HF token for speaker models
    if diarize:
        wx_device = os.getenv("WHISPER_DEVICE", "cpu")
        if wx_device == "mps":
            wx_device = "cpu"
        # Load ASR model for word alignment
        wx_asr_model = whisperx.load_model(model, device=wx_device, compute_type="float32")
        # Initialize diarization pipeline (HF token should be set in env HF_TOKEN)
        hf_token = os.getenv("HF_TOKEN", None)
        diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=wx_device)
        # Run diarization: pass original audio numpy array
        # and speaker count if provided and valid
        diarize_args = {}
        if num_speakers is not None and num_speakers > 0:
            diarize_args["num_speakers"] = num_speakers
            diarize_args["min_speakers"] = num_speakers
            diarize_args["max_speakers"] = num_speakers
        
        diarize_segments = diarization_pipeline(np_array, **diarize_args)
        # Assign speaker labels back to transcription result
        result = assign_word_speakers(diarize_segments, result)
        
        # Remap speaker labels to be 1-indexed
        if "segments" in result:
            for segment in result["segments"]:
                if "speaker" in segment and segment["speaker"].startswith("SPEAKER_"):
                    try:
                        speaker_id_str = segment["speaker"].split("_")[1]
                        speaker_id_int = int(speaker_id_str)
                        new_speaker_id_int = speaker_id_int + 1
                        segment["speaker"] = f"SPEAKER_{new_speaker_id_int:02d}"
                    except (IndexError, ValueError):
                        # If parsing fails, keep original label
                        pass
    return result


def convert_audio(file):
    """
    Converts audio received as an UploadFile to a numpy array
    compatible with openai-whisper. Adapted from `load_audio` in `whisper/audio.py`.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg"], capture_stdout=True, capture_stderr=True, input=file)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@app.post("/transcribe/text")
def transcribe_text(
    file: Annotated[UploadFile, File()],
    task: Annotated[str, Form()] = "transcribe",
    model: Annotated[str, Form()] = "base",
    initial_prompt: Annotated[str, Form()] = None,
    language: Annotated[str, Form()] = None,
    diarize: Annotated[bool, Form()] = False,
    num_speakers: Annotated[Optional[int], Form()] = None,
):
    bytes = file.file.read()
    np_array = convert_audio(bytes)
    whisper_instance = whisper.load_model(model)
    lang = None if not language or language.lower() == "string" else language
    result = whisper_instance.transcribe(
        audio=np_array,
        verbose=True,
        task=task,
        initial_prompt=initial_prompt,
        language=lang,
    )
    if diarize:
        wx_device = os.getenv("WHISPER_DEVICE", "cpu")
        if wx_device == "mps":
            wx_device = "cpu"
        wx_asr_model = whisperx.load_model(model, device=wx_device, compute_type="float32")
        hf_token = os.getenv("HF_TOKEN", None)
        diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=wx_device)
        
        diarize_args = {}
        if num_speakers is not None and num_speakers > 0:
            diarize_args["num_speakers"] = num_speakers
            diarize_args["min_speakers"] = num_speakers
            diarize_args["max_speakers"] = num_speakers
            
        diarize_segments = diarization_pipeline(np_array, **diarize_args)
        result = assign_word_speakers(diarize_segments, result)

        # Remap speaker labels to be 1-indexed
        if "segments" in result:
            for segment in result["segments"]:
                if "speaker" in segment and segment["speaker"].startswith("SPEAKER_"):
                    try:
                        speaker_id_str = segment["speaker"].split("_")[1]
                        speaker_id_int = int(speaker_id_str)
                        new_speaker_id_int = speaker_id_int + 1
                        segment["speaker"] = f"SPEAKER_{new_speaker_id_int:02d}"
                    except (IndexError, ValueError):
                        # If parsing fails, keep original label
                        pass

    # helper to format timestamps as [HH:MM:SS]
    def fmt_ts(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        td = timedelta(milliseconds=total_ms)
        hh = td.seconds // 3600
        mm = (td.seconds % 3600) // 60
        ss = td.seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    # build plain-text lines with start/end, speaker, and text
    lines = []
    for seg in result["segments"]:
        start = fmt_ts(seg["start"])
        end = fmt_ts(seg["end"])
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines)
