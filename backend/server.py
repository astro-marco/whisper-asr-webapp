import warnings
warnings.filterwarnings(
    "ignore",
    r"resource_tracker:.* leaked semaphore objects.*",
    category=UserWarning
)

from dotenv import load_dotenv
load_dotenv()  # load variables from .env file
import os
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Any
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
import tempfile
import torch
import time

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


def _resegment_diarized_result(result: Dict[str, Any], word_timestamps_requested: bool) -> List[Dict[str, Any]]:
    """
    Re-segments a transcription result based on word-level speaker assignments.
    Ensures each segment in the output corresponds to a single speaker.
    Handles words with no speaker assignment by attempting to associate them with the current speaker segment.
    Tags missing speakers as "NO_SPEAKER".
    """
    resegmented_list = []
    original_segments = result.get("segments", [])
    new_segment_id = 0

    for segment in original_segments:
        if "words" not in segment or not segment["words"]:
            # skip entirely empty segments (no words and no text)
            if not segment.get("text", "").strip():
                continue
            current_seg_copy = segment.copy()
            if "id" not in current_seg_copy:
                current_seg_copy["id"] = new_segment_id
                new_segment_id += 1
            current_seg_copy.setdefault("seek", segment.get("seek", 0))
            current_seg_copy.setdefault("tokens", segment.get("tokens", []))
            current_seg_copy.setdefault("temperature", segment.get("temperature", 0.0))
            current_seg_copy.setdefault("avg_logprob", segment.get("avg_logprob", 0.0))
            current_seg_copy.setdefault("compression_ratio", segment.get("compression_ratio", 0.0))
            current_seg_copy.setdefault("no_speech_prob", segment.get("no_speech_prob", 0.0))
            
            speaker_val = segment.get("speaker") or "NO_SPEAKER"
            current_seg_copy["speaker"] = speaker_val
            
            resegmented_list.append(current_seg_copy)
            continue

        current_sub_segment_word_details = []
        speaker_for_current_sub_segment = None

        for word_data in segment["words"]:
            original_speaker_of_word = word_data.get("speaker")
            if original_speaker_of_word == "": # Normalize empty string speaker from word_data
                original_speaker_of_word = None
            
            effective_speaker_for_this_word = original_speaker_of_word
            if effective_speaker_for_this_word is None:
                if current_sub_segment_word_details:
                    effective_speaker_for_this_word = speaker_for_current_sub_segment
                
            word_detail_to_add = {
                "word": word_data.get("word", ""),
                "start": word_data.get("start"),
                "end": word_data.get("end"),
                "speaker": effective_speaker_for_this_word 
            }

            if not current_sub_segment_word_details:
                current_sub_segment_word_details.append(word_detail_to_add)
                speaker_for_current_sub_segment = effective_speaker_for_this_word
            elif effective_speaker_for_this_word == speaker_for_current_sub_segment:
                current_sub_segment_word_details.append(word_detail_to_add)
            else: 
                if current_sub_segment_word_details:
                    seg_text = "".join(w["word"] for w in current_sub_segment_word_details).strip()
                    if seg_text:
                        chosen_speaker = speaker_for_current_sub_segment or "NO_SPEAKER"
                        new_seg_data = {
                            "id": new_segment_id, "seek": 0,
                            "start": current_sub_segment_word_details[0]["start"],
                            "end": current_sub_segment_word_details[-1]["end"],
                            "text": seg_text,
                            "speaker": chosen_speaker,
                            "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                            "compression_ratio": 0.0, "no_speech_prob": 0.0,
                        }
                        if word_timestamps_requested:
                            new_seg_data["words"] = [{"word": w["word"], "start": w["start"], "end": w["end"], "speaker": w["speaker"] or "NO_SPEAKER"} for w in current_sub_segment_word_details]
                        resegmented_list.append(new_seg_data)
                        new_segment_id += 1
                
                current_sub_segment_word_details = [word_detail_to_add]
                speaker_for_current_sub_segment = effective_speaker_for_this_word
        
        if current_sub_segment_word_details: 
            seg_text = "".join(w["word"] for w in current_sub_segment_word_details).strip()
            if seg_text:
                chosen_speaker = speaker_for_current_sub_segment or "NO_SPEAKER"
                new_seg_data = {
                    "id": new_segment_id, "seek": 0,
                    "start": current_sub_segment_word_details[0]["start"],
                    "end": current_sub_segment_word_details[-1]["end"],
                    "text": seg_text,
                    "speaker": chosen_speaker,
                    "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                    "compression_ratio": 0.0, "no_speech_prob": 0.0,
                }
                if word_timestamps_requested:
                    new_seg_data["words"] = [{"word": w["word"], "start": w["start"], "end": w["end"], "speaker": w["speaker"] or "NO_SPEAKER"} for w in current_sub_segment_word_details]
                resegmented_list.append(new_seg_data)
                new_segment_id += 1
            
    return resegmented_list


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
    # Save uploaded file to a temporary file for diarization
    tmp_audio_path = None
    if diarize:
        # Preserve original extension for ffmpeg compatibility
        suffix = Path(file.filename).suffix or ".wav"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(bytes)
        tmp_file.flush()
        tmp_file.close()
        tmp_audio_path = tmp_file.name
    # --- DEBUG: determina input per Whisper ---
    if tmp_audio_path:
        # uso il file .wav temporaneo
        audio_input = tmp_audio_path
    else:
        np_array = convert_audio(bytes)
        # debug rapido: controlla che l'array non sia vuoto
        print(f"[DEBUG] np_array shape={np_array.shape}, dtype={np_array.dtype}, min={np_array.min()}, max={np_array.max()}")
        audio_input = np_array
    whisper_instance = whisper.load_model(model)
    lang = None if not language or language.lower() == "string" else language

    effective_word_timestamps_for_process = word_timestamps
    if diarize:
        effective_word_timestamps_for_process = True

    print("Starting transcription...")
    start_trans_time = time.time()
    result = whisper_instance.transcribe(
        audio=audio_input,
        verbose=True,
        task=task,
        initial_prompt=initial_prompt,
        word_timestamps=effective_word_timestamps_for_process,
        language=lang,
    )
    end_trans_time = time.time()
    print(f"Transcription completed in {end_trans_time - start_trans_time:.2f}s")
    print(f"Transcript snippet: {result.get('text','')[:200]}")

    if diarize:
        wx_device = os.getenv("WHISPER_DEVICE", "cpu")
        if wx_device == "mps":
            wx_device = "cpu"
        
        hf_token = os.getenv("HF_TOKEN", None)
        diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=wx_device)
        print("Starting diarization...")
        start_diar_time = time.time()
        diarize_args = {}
        if num_speakers is not None and num_speakers > 0:
            diarize_args["num_speakers"] = num_speakers
            diarize_args["min_speakers"] = num_speakers
            diarize_args["max_speakers"] = num_speakers
        
        # Passa l'array NumPy direttamente alla pipeline di diarizzazione
        diarize_segments_raw = diarization_pipeline(tmp_audio_path, **diarize_args)
        result_with_word_speakers = assign_word_speakers(diarize_segments_raw, result)
        end_diar_time = time.time()
        print(f"Diarization completed in {end_diar_time - start_diar_time:.2f}s")
        
        if "segments" in result_with_word_speakers:
            for segment in result_with_word_speakers["segments"]:
                if "words" in segment and segment["words"]:
                    for word_info in segment["words"]:
                        if "speaker" in word_info and word_info["speaker"].startswith("SPEAKER_"):
                            try:
                                speaker_id_str = word_info["speaker"].split("_")[1]
                                speaker_id_int = int(speaker_id_str)
                                new_speaker_id_int = speaker_id_int + 1
                                word_info["speaker"] = f"SPEAKER_{new_speaker_id_int:02d}"
                            except (IndexError, ValueError):
                                pass
        
        result["segments"] = _resegment_diarized_result(result_with_word_speakers, word_timestamps)
            
    return result


def convert_audio(file):
    """
    Converts audio received as an UploadFile to a numpy array
    compatible with openai-whisper. Adapted from `load_audio` in `whisper/audio.py`.
    """
    try:
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

    effective_word_timestamps_for_process = True if diarize else False

    result = whisper_instance.transcribe(
        audio=np_array,
        verbose=True,
        task=task,
        initial_prompt=initial_prompt,
        word_timestamps=effective_word_timestamps_for_process,
        language=lang,
    )

    if diarize:
        wx_device = os.getenv("WHISPER_DEVICE", "cpu")
        if wx_device == "mps":
            wx_device = "cpu"
        
        hf_token = os.getenv("HF_TOKEN", None)
        diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=wx_device)
        diarize_args = {}
        if num_speakers is not None and num_speakers > 0:
            diarize_args["num_speakers"] = num_speakers
            diarize_args["min_speakers"] = num_speakers
            diarize_args["max_speakers"] = num_speakers
            
        # Passa l'array NumPy direttamente alla pipeline di diarizzazione
        diarize_segments_raw = diarization_pipeline(np_array, **diarize_args)
        result_with_word_speakers = assign_word_speakers(diarize_segments_raw, result)

        if "segments" in result_with_word_speakers:
            for segment in result_with_word_speakers["segments"]:
                if "words" in segment and segment["words"]:
                    for word_info in segment["words"]:
                        if "speaker" in word_info and word_info["speaker"].startswith("SPEAKER_"):
                            try:
                                speaker_id_str = word_info["speaker"].split("_")[1]
                                speaker_id_int = int(speaker_id_str)
                                new_speaker_id_int = speaker_id_int + 1
                                word_info["speaker"] = f"SPEAKER_{new_speaker_id_int:02d}"
                            except (IndexError, ValueError):
                                pass
        
        result["segments"] = _resegment_diarized_result(result_with_word_speakers, False)

    def fmt_ts(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        td = timedelta(milliseconds=total_ms)
        hh = td.seconds // 3600
        mm = (td.seconds % 3600) // 60
        ss = td.seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    lines = []
    for seg in result["segments"]:
        start = fmt_ts(seg["start"])
        end = fmt_ts(seg["end"])
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines)
