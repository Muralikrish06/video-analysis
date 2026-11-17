"""
Langflow Custom Component — Video Analysis Pipeline (Single Wrapper)
====================================================================
This REPLACES the four previous components:
  • AudioTranscriptionComponent   (CC1)
  • FaceDetectionComponent        (CC2)
  • AttentionAnalysisComponent    (CC3)
  • LLMExtractionComponent        (CC4)

Wiring (after this change)
--------------------------
  [Language Selector] ──► [Video Analysis Pipeline] ──► [Display]

How it works
------------
1. Receives the language_bundle from the Language Selector (Step 0).
2. POSTs video_path + language info to your FastAPI server running app.py.
3. Returns the extracted_data dict so the Display component can render it.

Configuration
-------------
Set the API_BASE_URL input to point at the machine running api.py, e.g.:
  • Same machine : http://localhost:8000
  • Remote server: http://192.168.1.50:8000
  • Docker        : http://video-analysis-api:8000
"""

import requests
from langflow.custom import Component
from langflow.io import DataInput, StrInput, Output
from langflow.schema import Data


class VideoAnalysisPipelineComponent(Component):
    display_name = "📹 Video Analysis Pipeline"
    description  = (
        "Single component that replaces Transcription + Face Detection + "
        "Attention Analysis + LLM Extraction. Delegates all heavy compute to "
        "the FastAPI/LangGraph server (api.py)."
    )
    icon = "video"
    name = "VideoAnalysisPipelineComponent"

    # ── Inputs ────────────────────────────────────────────────────────────────
    inputs = [
        DataInput(
            name         = "language_bundle",
            display_name = "Language Bundle (from Language Selector)",
            info         = "Connect the output of the Language Selector component here.",
            required     = True,
        ),
        StrInput(
            name         = "api_base_url",
            display_name = "API Base URL",
            info         = (
                "Base URL of your running FastAPI server (api.py). "
                "Example: http://localhost:8000"
            ),
            value        = "http://localhost:8000",
            required     = True,
        ),
    ]

    # ── Outputs ───────────────────────────────────────────────────────────────
    outputs = [
        Output(
            display_name = "Analysis Result ▶",
            name         = "result",
            info         = "Connect to the Display component.",
            method       = "run_pipeline",
        ),
    ]

    # ── Logic ─────────────────────────────────────────────────────────────────
    def run_pipeline(self) -> Data:
        # Unpack the language bundle from upstream
        incoming = self.language_bundle
        payload  = incoming.data if isinstance(incoming, Data) else incoming

        video_path    = payload.get("video_path", "")
        language_name = payload.get("language_name", "English")
        language_code = payload.get("language_code", "en")
        base_url      = (self.api_base_url or "http://localhost:8000").rstrip("/")

        self.log(
            f"Calling pipeline API at {base_url}/analyze "
            f"| video={video_path} | language={language_name} ({language_code})"
        )

        if not video_path:
            raise ValueError("video_path is empty — check the Language Selector component.")

        # ── Call the FastAPI server ───────────────────────────────────────────
        try:
            response = requests.post(
                f"{base_url}/analyze",
                json={
                    "video_path":    video_path,
                    "language_name": language_name,
                    "language_code": language_code,
                },
                timeout=600,  # 10-minute timeout — Whisper + YOLO can be slow
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to the pipeline API at {base_url}. "
                "Make sure api.py is running: uvicorn api:app --host 0.0.0.0 --port 8000"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Pipeline API timed out after 10 minutes. "
                "The video may be too long, or the server is under heavy load."
            )
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Pipeline API returned an error: {exc.response.status_code} — "
                f"{exc.response.text[:300]}"
            )

        result = response.json()
        self.log(f"Pipeline complete. Face detected: {result.get('face_detected')}")

        # Forward everything downstream (Display component reads extracted_data)
        return Data(
            data={
                **payload,                                      # language metadata
                "extracted_data":  result.get("extracted_data", {}),
                "face_detected":   result.get("face_detected", False),
                "transcript":      result.get("transcript", ""),
                "pipeline_status": result.get("status", "complete"),
            }
        )