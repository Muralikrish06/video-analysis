"""
api.py — FastAPI wrapper around the LangGraph pipeline
=======================================================
Run this on your GPU/compute server (alongside app.py).

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

For multiple parallel videos (if you have enough VRAM/RAM):
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2

Endpoints
---------
POST /analyze        — run the full pipeline, return extracted JSON
GET  /health         — quick liveness check
GET  /languages      — list supported languages
"""

import os
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import from your existing app.py (must be in the same directory)
from app import run_pipeline, download_dlib_model, SUPPORTED_LANGUAGES, LOGGER
from config import settings


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download dlib model once when the server starts."""
    LOGGER.info("FastAPI server starting — downloading dlib model if needed.")
    download_dlib_model()
    LOGGER.info("Server ready.")
    yield
    LOGGER.info("Server shutting down.")


app = FastAPI(
    title="Video Analysis API",
    description="Wraps the LangGraph pipeline (Whisper → YOLOv8 → dlib → LLM) as a REST endpoint.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Langflow (running on any port/host) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your Langflow host in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    video_path    : absolute path to the video file on THIS server's filesystem.
    language_name : e.g. "English"
    language_code : e.g. "en"
    """
    video_path:    str
    language_name: str = "English"
    language_code: str = "en"


class AnalyzeResponse(BaseModel):
    status:         str
    extracted_data: dict
    face_detected:  bool
    transcript:     str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — returns 200 if the server is running."""
    return {"status": "ok"}


@app.get("/languages")
def languages():
    """Return the supported language list so Langflow can populate a dropdown."""
    return {
        key: {"name": name, "code": code}
        for key, (name, code) in SUPPORTED_LANGUAGES.items()
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Run the full LangGraph pipeline on a video file.

    The video_path must be accessible on THIS server's filesystem.
    If Langflow and the API server are on different machines, use
    the /upload-and-analyze endpoint instead.
    """
    LOGGER.info(
        "POST /analyze | video=%s language=%s (%s)",
        req.video_path, req.language_name, req.language_code
    )

    if not os.path.exists(req.video_path):
        LOGGER.error("Video file not found: %s", req.video_path)
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found on server: {req.video_path}"
        )

    try:
        final_state = run_pipeline(req.video_path, req.language_name, req.language_code)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return AnalyzeResponse(
        status        = "complete",
        extracted_data= final_state.get("extracted_data", {}),
        face_detected = final_state.get("face_detected", False),
        transcript    = final_state.get("transcript", ""),
    )


@app.post("/upload-and-analyze", response_model=AnalyzeResponse)
async def upload_and_analyze(
    video: UploadFile = File(...),
    language_name: str = "English",
    language_code: str = "en",
):
    """
    Upload a video file directly from Langflow (for when Langflow and the API
    server are on different machines).

    Saves the upload to /tmp, runs the pipeline, then deletes the temp file.
    """
    upload_dir = os.getenv("UPLOAD_DIR", settings.paths.upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

    tmp_path = os.path.join(upload_dir, video.filename)
    LOGGER.info("Receiving upload: %s", video.filename)

    try:
        contents = await video.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        LOGGER.info("Upload saved to: %s (%d bytes)", tmp_path, len(contents))

        final_state = run_pipeline(tmp_path, language_name, language_code)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            LOGGER.info("Temp file cleaned up: %s", tmp_path)

    return AnalyzeResponse(
        status        = "complete",
        extracted_data= final_state.get("extracted_data", {}),
        face_detected = final_state.get("face_detected", False),
        transcript    = final_state.get("transcript", ""),
    )