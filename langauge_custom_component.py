"""
Langflow Custom Component — Step 0: Language Selector
======================================================
This is the FIRST component in the pipeline.

Inputs
------
• video_file : FileInput  — browse and upload the video file directly from the UI
• language   : dropdown   — English / Tamil / Kannada / Malayalam / Hindi / Marathi / Telugu

Outputs
-------
• to_transcription  : Data — forwarded to AudioTranscriptionComponent (CC1)
                             Always fires. CC1 reads language_code internally
                             and decides whether to run Whisper or skip.

Wiring
------
  [Language Selector] ──to_transcription──► [Audio Transcription (CC1)]
                                                     │
                                               (rest of pipeline unchanged)

The language_bundle dict that travels through the whole pipeline contains:
  video_path, language_name, language_code,
  transcription_skipped, whisper_enabled, transcript_note
"""

from langflow.custom import Component
from langflow.io import DropdownInput, FileInput, Output
from langflow.schema import Data


SUPPORTED_LANGUAGES: dict[str, tuple[str, str]] = {
    "English":   ("English",   "en"),
    "Tamil":     ("Tamil",     "ta"),
    "Kannada":   ("Kannada",   "kn"),
    "Malayalam": ("Malayalam", "ml"),
    "Hindi":     ("Hindi",     "hi"),
    "Marathi":   ("Marathi",   "mr"),
    "Telugu":    ("Telugu",    "te"),
}

# Accepted video file extensions shown in the file browser
VIDEO_FILE_TYPES = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v", "mpeg", "mpg"]


class LanguageSelectorComponent(Component):
    display_name = "🌐 Language Selector"
    description  = (
        "Step 0 — Browse and upload a video file, then select the spoken language. "
        "Routes to Whisper for English; skips transcription for all other languages. "
        "Connect the single output to the Audio Transcription component."
    )
    icon = "languages"
    name = "LanguageSelectorComponent"

    # ── Inputs ────────────────────────────────────────────────────────────────
    inputs = [
        FileInput(
            name         = "video_file",
            display_name = "Video File",
            info         = (
                "Click to browse your system and select a video file "
                "(mp4, avi, mov, mkv, wmv, flv, webm, m4v, mpeg, mpg). "
                "The resolved file path is forwarded through the pipeline."
            ),
            file_types   = VIDEO_FILE_TYPES,
            required     = True,
        ),
        DropdownInput(
            name         = "language",
            display_name = "Video Language",
            info         = "Select the language spoken in the video.",
            options      = list(SUPPORTED_LANGUAGES.keys()),
            value        = "English",
            required     = True,
        ),
    ]

    # ── Outputs ───────────────────────────────────────────────────────────────
    outputs = [
        Output(
            display_name = "To Transcription ▶",
            name         = "to_transcription",
            info         = "Connect to Audio Transcription (CC1). Always fires.",
            method       = "build_bundle",
        ),
    ]

    # ── Output method ─────────────────────────────────────────────────────────
    def build_bundle(self) -> Data:
        # FileInput resolves to the absolute path of the uploaded/selected file
        video_path = (self.video_file or "").strip()
        lang_key   = self.language or "English"

        language_name, language_code = SUPPORTED_LANGUAGES.get(
            lang_key, ("English", "en")
        )

        is_english = language_code == "en"

        if is_english:
            self.log(
                f"Language: {language_name} ({language_code}) — "
                "Whisper transcription ENABLED."
            )
        else:
            self.log(
                f"Language: {language_name} ({language_code}) — "
                "Transcription SKIPPED (no model available yet). "
                "Pipeline will proceed directly from transcription stub → "
                "Face Detection → (Attention if face found) → LLM Extraction."
            )

        self.log(f"Video path: {video_path}")

        transcript_note = (
            ""
            if is_english
            else (
                f"The video language is '{language_name}' ({language_code}). "
                "Transcription model is not yet available for this language. "
                "Transcript is unavailable — set all transcript-derived fields to N/A."
            )
        )

        return Data(
            data={
                "video_path":            video_path,
                "language_name":         language_name,
                "language_code":         language_code,
                "transcription_skipped": not is_english,
                "whisper_enabled":       is_english,
                "transcript_note":       transcript_note,
                # downstream components read these keys directly
                "transcript":            "",          # filled by CC1 if English
                "status":                "language_selected",
            }
        )