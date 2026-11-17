"""
Video Analysis Pipeline with LangGraph.
Logging is handled via the project-standard rotating-file LOGGER.
"""

import json
import operator
import os
import logging
import subprocess
import traceback
from logging.handlers import RotatingFileHandler
from typing import Annotated, TypedDict

import cv2
import dlib
import numpy as np
import requests
import torch
from imutils import face_utils
from langgraph.graph import END, START, StateGraph
from scipy.spatial import distance as dist

from config import settings

try:
    import moviepy.editor as mp
except ImportError:
    subprocess.run(["pip", "install", "moviepy"], check=True)
    import moviepy.editor as mp

try:
    import whisper
except ImportError:
    subprocess.run(["pip", "install", "openai-whisper"], check=True)
    import whisper

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    subprocess.run(["pip", "install", "ultralytics"], check=True)
    import ultralytics
    from ultralytics import YOLO

import ultralytics.nn.modules
import ultralytics.nn.modules.block
import ultralytics.nn.tasks
import torch.nn.modules.activation
import torch.nn.modules.batchnorm
from datetime import datetime


# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logger(logger_name, log_file, log_level_env, backup_count=10, max_bytes=20000000):
    """Setup a logger with rotating file handler."""
    log_dir = os.getenv('LOG_DIRECTORY', './')
    logger_file_path = os.path.join(str(log_dir), log_file)

    log_level_dict = {
        'DEBUG':    logging.DEBUG,
        'INFO':     logging.INFO,
        'WARNING':  logging.WARNING,
        'ERROR':    logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    log_level = log_level_dict.get(log_level_env, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handler = RotatingFileHandler(logger_file_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s] [ %(lineno)s:%(funcName)5s()] %(message)s',
        datefmt='%Y-%b-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger, log_level


log_level_env = os.getenv('LOG_LEVEL', settings.logging.level)
LOGGER, LOG_LEVEL = setup_logger(
    'AI_TEXT_PROCESSING_LOGGER',
    settings.logging.text_log_file,
    log_level_env,
    backup_count=settings.logging.backup_count,
    max_bytes=settings.logging.max_bytes,
)

metrics_log_level_env = os.getenv('METRICS_LOG_LEVEL', settings.logging.metrics_level)
METRICS_LOGGER, METRICS_LOG_LEVEL = setup_logger(
    'METRICS_LOGGER',
    settings.logging.metrics_log_file,
    metrics_log_level_env,
    backup_count=settings.logging.backup_count,
    max_bytes=settings.logging.max_bytes,
)

# ──────────────────────────────────────────────────────────────────────────────


# ─── LLM Extraction Config ────────────────────────────────────────────────────

LLM_API_URL = settings.llm.api_url
LLM_MODEL   = settings.llm.model

# ─── Language Configuration ───────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    key: (val["name"], val["code"])
    for key, val in settings.languages.items()
}


def select_language() -> tuple[str, str]:
    """Prompt the user to pick a language. Returns (language_name, language_code)."""
    LOGGER.info("Prompting user for language selection.")
    for key, (name, code) in SUPPORTED_LANGUAGES.items():
        print(f"    [{key}]  {name}")

    while True:
        try:
            choice = input("  Enter choice (1–7): ").strip()
        except EOFError:
            LOGGER.warning("EOF received during language selection — defaulting to English.")
            print("  No input received — defaulting to English.")
            return "English", "en"

        if choice in SUPPORTED_LANGUAGES:
            name, code = SUPPORTED_LANGUAGES[choice]
            LOGGER.info("Language selected: %s (%s)", name, code)
            print(f"\n  ✅ Language selected: {name} ({code})")
            if code != "en":
                LOGGER.warning(
                    "Transcription model not available for language '%s' (%s). "
                    "Audio transcription will be skipped.", name, code
                )
                print(f"  ⚠  Transcription model for '{name}' is not available yet.")
                print("     Audio transcription will be SKIPPED for this session.")
            return name, code
        LOGGER.warning("Invalid language choice entered: '%s'", choice)
        print("  ❌ Invalid choice. Please enter a number between 1 and 7.")


# ─── Extraction Schema ────────────────────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "customer_name":                  "Full name of the customer/applicant",
    "policy_name":                    "Name of the insurance policy product",
    "company_name":                   "Name of the insurance company",
    "policy_term_years":              "Total duration/term of the policy in years",
    "premium_amount":                 "Premium amount (with currency symbol if present)",
    "premium_payment_years":          "Number of years the premium needs to be paid",
    "premium_payment_frequency":      "Payment frequency (annual/monthly/quarterly etc.)",
    "withdrawal_allowed_after_years": "Number of policy years after which withdrawals are allowed",
    "free_look_period_days":          "Free look / free cancellation period in days",
    "consent_given":                  "Whether consent was given by the customer (Yes/No)",
    "consent_statement":              "The exact consent statement read aloud by the customer",
    "face_detected":                  "Whether a face was detected in the video (True/False)",
    "attention_score":                "Overall attention score (out of 100)",
    "attention_grade":                "Attention grade (A/B/C/D/F)",
    "attention_verdict":              "Attention verdict/description",
    "focused_frames_pct":             "Percentage of frames where attention was focused (≥75)",
    "distracted_frames_pct":          "Percentage of frames where attention was distracted (<60)",
    "blink_count":                    "Total number of blinks detected",
    "blink_rate_per_min":             "Blink rate per minute",
    "gaze_stability":                 "Gaze stability score (out of 100)",
    "focus_score":                    "Focus/sharpness score (out of 100)",
    "no_face_frames":                 "Number of frames with no face detected",
    "phone_detected":                 "Whether a phone was detected during the session (Yes/No)",
    "output_video_path":              "Path to the attention analysis output video",
    "is_fixed_deposit":               "Whether this is described as a fixed deposit (Yes/No)",
    "is_mutual_fund":                 "Whether this is described as a mutual fund (Yes/No)",
    "is_single_premium":              "Whether this is a single premium policy (Yes/No)",
    "data_consent_to_insurer":        "Whether customer consented to share data with insurer (Yes/No)",
}


# ─── Pipeline State ───────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    video_path:              str
    language_name:           str
    language_code:           str
    transcript:              Annotated[str, operator.add]
    face_detected:           bool
    detected_objects:        dict
    face_image_path:         Annotated[str, operator.add]
    yolo_output_video:       Annotated[str, operator.add]
    face_detection_report:   Annotated[str, operator.add]
    attention_report:        Annotated[str, operator.add]
    attention_output_video:  Annotated[str, operator.add]
    extracted_data:          dict
    extraction_report:       Annotated[str, operator.add]


# ─── Node 1 — Audio Transcription (Whisper) ───────────────────────────────────

class VideoTranscriber:
    DEFAULT_MODEL = os.getenv("WHISPER_MODEL", settings.whisper.model)

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            LOGGER.error("Video file not found: %s", video_path)
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path
        self.audio_path = "extracted_audio.mp3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        LOGGER.info("VideoTranscriber initialised. Device: %s | Video: %s", self.device, video_path)

    def _extract_audio(self):
        LOGGER.info("Extracting audio from video: %s", self.video_path)
        clip = mp.VideoFileClip(self.video_path)
        clip.audio.write_audiofile(self.audio_path, logger=None)
        clip.audio.close()
        clip.close()
        LOGGER.info("Audio saved to: %s", self.audio_path)

    def _load_model(self, model_name: str):
        LOGGER.info("Loading Whisper model '%s' on %s", model_name, self.device)
        self.model = whisper.load_model(model_name, device=self.device)
        LOGGER.info("Whisper model loaded successfully.")

    def transcribe(self, model_name: str | None = None) -> str:
        model_name = model_name or self.DEFAULT_MODEL
        LOGGER.info("Starting transcription. Model: %s", model_name)
        self._extract_audio()
        self._load_model(model_name)
        result = self.model.transcribe(self.audio_path)
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)
            LOGGER.debug("Temporary audio file removed: %s", self.audio_path)
        LOGGER.info("Transcription complete. Characters: %d", len(result["text"]))
        return result["text"]


def transcribe_audio_node(state: PipelineState) -> dict:
    LOGGER.info("NODE 1 started: Audio Transcription.")

    lang_name = state.get("language_name", "English")
    lang_code = state.get("language_code", "en")

    if lang_code != "en":
        LOGGER.warning(
            "Skipping transcription — '%s' (%s) model not available.", lang_name, lang_code
        )
        LOGGER.info("NODE 1 complete: transcript set to empty (N/A).")
        return {"transcript": ""}

    try:
        transcriber = VideoTranscriber(state["video_path"])
        transcript = transcriber.transcribe()

        # ── CHANGE 1: Print the full transcript to stdout immediately after transcription ──
        _print_full_transcript(transcript, lang_name)

        LOGGER.info("NODE 1 complete: transcription successful.")
        return {"transcript": transcript}
    except Exception as exc:
        LOGGER.error("NODE 1 failed: %s\n%s", exc, traceback.format_exc())
        raise


def _print_full_transcript(transcript: str, lang_name: str = "English") -> None:
    """Pretty-print the full transcript to stdout and log it."""
    separator = "═" * 58
    header = f"""
{separator}
         FULL TRANSCRIPT  [{lang_name}]
{separator}
{transcript.strip()}
{separator}
"""
    print(header)
    LOGGER.info("FULL TRANSCRIPT:\n%s", transcript)


# ─── Node 2 — Face Detection (YOLOv8) ────────────────────────────────────────

class VideoAnalyzer:
    def __init__(self, video_path: str, model_path: str = settings.yolo.model_path,
                 start_sec: int = settings.yolo.start_sec, end_sec: int = settings.yolo.end_sec):
        self.video_path       = video_path
        self.model_path       = model_path
        self.start_sec        = start_sec
        self.end_sec          = end_sec
        self.output_video     = "yolo_detected.mp4"
        self.face_output_dir  = "captured_faces"
        self.model            = None
        self.cap              = None
        self.out              = None
        self.fps = self.frame_width = self.frame_height = 0
        self.detected_objects  = {}
        self.face_detected     = False
        self.last_face_crop    = None
        self.last_face_conf    = 0.0
        self.last_face_frame   = 0
        os.makedirs(self.face_output_dir, exist_ok=True)
        LOGGER.info(
            "VideoAnalyzer initialised. Video: %s | Range: %ds–%ds",
            video_path, start_sec, end_sec
        )

    def _load_model(self):
        LOGGER.info("Loading YOLO model: %s", self.model_path)
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.DetectionModel,
            torch.nn.modules.container.Sequential,
            ultralytics.nn.modules.conv.Conv,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            ultralytics.nn.modules.block.C2f,
        ])
        self.model = YOLO(self.model_path)
        LOGGER.info("YOLO model loaded successfully.")

    def _setup_video(self):
        LOGGER.info("Opening video for YOLO analysis: %s", self.video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            LOGGER.error("Cannot open video: %s", self.video_path)
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        self.fps          = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_sec * self.fps)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_video, fourcc, self.fps,
            (self.frame_width, self.frame_height)
        )
        LOGGER.info("Video setup: %dx%d @ %d fps", self.frame_width, self.frame_height, self.fps)

    def _process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        for result in self.model(frame):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf     = float(box.conf[0])
                cls_name = self.model.names[int(box.cls[0])]
                if cls_name not in self.detected_objects or conf > self.detected_objects[cls_name]:
                    self.detected_objects[cls_name] = conf
                color = (0, 255, 0)
                if cls_name == "person":
                    self.face_detected = True
                    color = (0, 0, 255)
                    pad  = 20
                    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                    px2 = min(self.frame_width,  x2 + pad)
                    py2 = min(self.frame_height, y2 + pad)
                    self.last_face_crop  = frame[py1:py2, px1:px2].copy()
                    self.last_face_conf  = conf
                    self.last_face_frame = frame_count
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _save_face(self) -> str | None:
        if self.last_face_crop is None or self.last_face_crop.size == 0:
            LOGGER.warning("No face crop available to save.")
            return None
        ts   = datetime.now().strftime("%H%M%S_%f")
        name = f"face_f{self.last_face_frame}_c{self.last_face_conf:.2f}_{ts}.jpg"
        path = os.path.join(self.face_output_dir, name)
        cv2.imwrite(path, self.last_face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        LOGGER.info("Face image saved: %s", path)
        return path

    def run(self) -> tuple[bool, dict, str | None]:
        LOGGER.info("Starting YOLO face-detection run.")
        self._load_model()
        self._setup_video()
        end_frame   = self.end_sec   * self.fps
        start_frame = self.start_sec * self.fps
        frame_count = start_frame
        LOGGER.info("Analysing frames %d → %d", start_frame, end_frame)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame_count >= end_frame:
                break
            frame_count += 1
            self.out.write(self._process_frame(frame, frame_count))
        self.cap.release()
        self.out.release()
        face_path = self._save_face() if self.face_detected else None
        LOGGER.info(
            "YOLO run complete. Face detected: %s | Objects: %s",
            self.face_detected, self.detected_objects
        )
        return self.face_detected, self.detected_objects, face_path

    def summary_report(self) -> str:
        lines = ["\n═══ YOLO Face-Detection Report ═══"]
        lines.append(f"  Face detected : {'YES' if self.face_detected else 'NO'}")
        for obj, conf in sorted(self.detected_objects.items(), key=lambda x: -x[1]):
            bar = "█" * int(conf * 20)
            lines.append(f"  {obj:<20} {conf:.2f}  {bar}")
        lines.append(f"  Output        : {self.output_video}")
        LOGGER.debug("YOLO summary report generated.")
        return "\n".join(lines)


def detect_faces_node(state: PipelineState) -> dict:
    LOGGER.info("NODE 2 started: Face Detection (YOLOv8).")
    try:
        analyzer = VideoAnalyzer(
            video_path=state["video_path"],
            start_sec=0,
            end_sec=5,
        )
        face_detected, detected_objects, face_path = analyzer.run()
        report = analyzer.summary_report()
        LOGGER.info(
            "NODE 2 complete. Face detected: %s | Objects: %s",
            face_detected, detected_objects
        )
        return {
            "face_detected":         face_detected,
            "detected_objects":      detected_objects,
            "face_image_path":       face_path or "",
            "yolo_output_video":     analyzer.output_video,
            "face_detection_report": report,
        }
    except Exception as exc:
        LOGGER.error("NODE 2 failed: %s\n%s", exc, traceback.format_exc())
        raise


def route_after_face_detection(state: PipelineState) -> str:
    if state.get("face_detected"):
        LOGGER.info("Routing: face detected → analyze_attention")
        return "analyze_attention"
    LOGGER.warning("Routing: no face detected → skipping to extract_data")
    return "extract_data"


# ─── Node 3 — Attention Score Analysis (dlib) ────────────────────────────────

class AttentionScoreAnalyzer:
    def __init__(self, video_path: str, start_sec: int = settings.attention.start_sec,
                 end_sec: int | None = settings.attention.end_sec,
                 detected_objects: dict = None):
        self.video_path        = video_path
        self.start_sec         = start_sec
        self.end_sec           = end_sec
        self.detected_objects  = detected_objects or {}
        self.PHONE_PENALTY     = settings.attention.phone_penalty
        self.output_dir        = settings.attention.output_dir
        self.output_video      = os.path.join(self.output_dir, "attention_output.mp4")
        self.detector          = None
        self.predictor         = None
        self.cap               = None
        self.out               = None
        self.fps = self.frame_width = self.frame_height = 0
        self.LEFT_EYE_IDX      = list(range(36, 42))
        self.RIGHT_EYE_IDX     = list(range(42, 48))
        self.NOSE_IDX          = 30
        self.CHIN_IDX          = 8
        self.LEFT_TEMPLE_IDX   = 0
        self.RIGHT_TEMPLE_IDX  = 16
        self.attention_scores  = []
        self.ear_values        = []
        self.gaze_h_values     = []
        self.gaze_v_values     = []
        self.head_yaw_values   = []
        self.head_pitch_values = []
        self.focus_scores      = []
        self.distraction_frames = []
        self.blink_count       = 0
        self.blink_in_progress = False
        self.no_face_frames    = 0
        self.EAR_BLINK_THRESH  = settings.attention.thresholds.ear_blink
        self.GAZE_LEFT_THRESH  = settings.attention.thresholds.gaze_left
        self.GAZE_RIGHT_THRESH = settings.attention.thresholds.gaze_right
        self.GAZE_UP_THRESH    = settings.attention.thresholds.gaze_up
        self.GAZE_DOWN_THRESH  = settings.attention.thresholds.gaze_down
        self.YAW_THRESH        = settings.attention.thresholds.yaw
        self.PITCH_THRESH      = settings.attention.thresholds.pitch
        os.makedirs(self.output_dir, exist_ok=True)
        LOGGER.info(
            "AttentionScoreAnalyzer initialised. Video: %s | Range: %ds–%s",
            video_path, start_sec, end_sec or "end"
        )

    def setup(self):
        model_path = settings.dlib.model_path
        if not os.path.exists(model_path):
            LOGGER.error("dlib landmark model not found: %s", model_path)
            raise FileNotFoundError(
                "dlib model not found. Run the following commands first:\n"
                "  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"
            )
        LOGGER.info("Loading dlib detector and predictor.")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.cap       = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            LOGGER.error("Cannot open video for attention analysis: %s", self.video_path)
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        self.fps          = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.end_sec is None:
            self.end_sec = total_frames // self.fps
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_sec * self.fps)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_video, fourcc, self.fps,
            (self.frame_width, self.frame_height)
        )
        LOGGER.info(
            "Attention video setup: %dx%d @ %d fps | %ds–%ds",
            self.frame_width, self.frame_height, self.fps, self.start_sec, self.end_sec
        )

    def _ear(self, pts):
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    def _gaze(self, gray, eye_pts):
        x_c, y_c = eye_pts[:, 0], eye_pts[:, 1]
        x1, x2   = int(np.min(x_c)) - 5, int(np.max(x_c)) + 5
        y1, y2   = int(np.min(y_c)) - 5, int(np.max(y_c)) + 5
        x1, y1   = max(0, x1), max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.5, 0.5
        _, thresh   = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5, 0.5
        pupil = max(contours, key=cv2.contourArea)
        M     = cv2.moments(pupil)
        if M["m00"] == 0:
            return 0.5, 0.5
        cx = float(np.clip(M["m10"] / M["m00"] / (x2 - x1 + 1e-6), 0, 1))
        cy = float(np.clip(M["m01"] / M["m00"] / (y2 - y1 + 1e-6), 0, 1))
        return cx, cy

    def _head_pose(self, shape):
        nose  = shape[self.NOSE_IDX]
        chin  = shape[self.CHIN_IDX]
        lt    = shape[self.LEFT_TEMPLE_IDX]
        rt    = shape[self.RIGHT_TEMPLE_IDX]
        fw    = np.linalg.norm(rt - lt) + 1e-6
        cx    = (lt[0] + rt[0]) / 2
        cy    = (nose[1] + chin[1]) / 2
        return float((nose[0] - cx) / fw), float((nose[1] - cy) / fw)

    def _focus(self, gray, rect):
        x1, y1 = max(0, rect.left()), max(0, rect.top())
        x2 = min(gray.shape[1], rect.right())
        y2 = min(gray.shape[0], rect.bottom())
        roi = gray[y1:y2, x1:x2]
        return float(min(100.0, cv2.Laplacian(roi, cv2.CV_64F).var() / 3.0)) if roi.size else 0.0

    def _attention(self, gaze_h, gaze_v, ear, yaw, pitch, focus, is_new_blink=False):
        score, reasons = 100.0, []
        if gaze_h < self.GAZE_LEFT_THRESH:
            score -= (self.GAZE_LEFT_THRESH - gaze_h) / self.GAZE_LEFT_THRESH * 30
            reasons.append("Looking LEFT")
        elif gaze_h > self.GAZE_RIGHT_THRESH:
            score -= (gaze_h - self.GAZE_RIGHT_THRESH) / (1 - self.GAZE_RIGHT_THRESH) * 30
            reasons.append("Looking RIGHT")
        if gaze_v < self.GAZE_UP_THRESH:
            score -= (self.GAZE_UP_THRESH - gaze_v) / self.GAZE_UP_THRESH * 15
            reasons.append("Looking UP")
        elif gaze_v > self.GAZE_DOWN_THRESH:
            score -= (gaze_v - self.GAZE_DOWN_THRESH) / (1 - self.GAZE_DOWN_THRESH) * 15
            reasons.append("Looking DOWN")
        if is_new_blink:
            score -= 20
            reasons.append("Eyes closed")
        if abs(yaw) > self.YAW_THRESH:
            score -= (abs(yaw) - self.YAW_THRESH) / (1 - self.YAW_THRESH) * 25
            reasons.append("Head turned")
        if abs(pitch) > self.PITCH_THRESH:
            score -= (abs(pitch) - self.PITCH_THRESH) / (1 - self.PITCH_THRESH) * 15
            reasons.append("Head tilted")
        if focus < 30:
            score -= (30 - focus) * 0.5
            reasons.append("Low focus")
        if "cell phone" in self.detected_objects:
            score -= self.PHONE_PENALTY
            reasons.append("Phone detected")
        return round(max(0.0, min(100.0, score)), 2), reasons

    def _overlay(self, frame, score, gaze_h, gaze_v, ear, yaw, pitch, focus, reasons,
                 avg_score, grade, verdict, focused_pct, distracted_pct, blink_rate):
        h, w = frame.shape[:2]
        left_panel_w = 360
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (left_panel_w, 310), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        color = (0, 255, 0) if score >= 75 else (0, 165, 255) if score >= 50 else (0, 0, 255)
        cv2.putText(frame, f"Attention: {score:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"Gaze H:{gaze_h:.2f} V:{gaze_v:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"EAR:{ear:.2f}  Blinks:{self.blink_count}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Yaw:{yaw:.2f}  Pitch:{pitch:.2f}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Focus:{focus:.1f}",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y = 165
        for r in reasons[:3]:
            cv2.putText(frame, f">> {r}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            y += 22
        bx, by = 10, h - 30
        bw, bh = w - 20, 18
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * score / 100), by + bh), color, -1)
        cv2.putText(frame, f"{score:.0f}%",
                    (bx + bw // 2 - 15, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panel_x  = w - 300
        panel_w  = 290
        panel_h  = 280
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x - 5, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        grade_color = (
            (0, 255, 0)   if grade == "A" else
            (100, 220, 0) if grade == "B" else
            (0, 165, 255) if grade == "C" else
            (0, 80, 255)  if grade == "D" else
            (0, 0, 255)
        )
        summary_lines = [
            ("SUMMARY REPORT", (255, 220, 0), 0.55, 2),
            (f"Avg Score : {avg_score:.1f}%  [{grade}]", grade_color, 0.5, 1),
            (verdict,                                      (200, 200, 200), 0.42, 1),
            (f"Focused   : {focused_pct:.1f}%",           (0, 200, 100),   0.42, 1),
            (f"Distracted: {distracted_pct:.1f}%",        (0, 80, 255),    0.42, 1),
            (f"Blink Rate: {blink_rate:.1f}/min",         (200, 200, 200), 0.42, 1),
            (f"No-face   : {self.no_face_frames} fr",     (180, 180, 180), 0.42, 1),
        ]
        ry = 22
        for text, col, scale, thickness in summary_lines:
            cv2.putText(frame, text, (panel_x, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness)
            ry += int(scale * 60)
        bar_y  = ry + 8
        bar_bw = panel_w - 10
        cv2.rectangle(frame, (panel_x, bar_y), (panel_x + bar_bw, bar_y + 16), (50, 50, 50), -1)
        cv2.rectangle(frame, (panel_x, bar_y),
                      (panel_x + int(bar_bw * avg_score / 100), bar_y + 16), grade_color, -1)
        cv2.putText(frame, f"Avg {avg_score:.0f}%",
                    (panel_x + bar_bw // 2 - 22, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    def _process_frame(self, frame, idx):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if not rects:
            self.no_face_frames += 1
            self.attention_scores.append(20.0)
            self.distraction_frames.append(idx)
            cv2.putText(frame, "No Face", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            LOGGER.debug("Frame %d: no face detected.", idx)
            return frame

        rect   = rects[0]
        shape  = face_utils.shape_to_np(self.predictor(gray, rect))
        le, re = shape[self.LEFT_EYE_IDX], shape[self.RIGHT_EYE_IDX]
        ear    = (self._ear(le) + self._ear(re)) / 2.0
        gh_l, gv_l = self._gaze(gray, le)
        gh_r, gv_r = self._gaze(gray, re)
        gaze_h, gaze_v = (gh_l + gh_r) / 2.0, (gv_l + gv_r) / 2.0
        yaw, pitch = self._head_pose(shape)
        focus      = self._focus(gray, rect)

        is_new_blink = ear < self.EAR_BLINK_THRESH and not self.blink_in_progress
        if ear < self.EAR_BLINK_THRESH:
            if not self.blink_in_progress:
                self.blink_count += 1
                self.blink_in_progress = True
                LOGGER.debug("Frame %d: blink detected. Total blinks: %d", idx, self.blink_count)
        else:
            self.blink_in_progress = False

        score, reasons = self._attention(gaze_h, gaze_v, ear, yaw, pitch, focus, is_new_blink)
        self.attention_scores.append(score)
        self.ear_values.append(ear)
        self.gaze_h_values.append(gaze_h)
        self.gaze_v_values.append(gaze_v)
        self.head_yaw_values.append(yaw)
        self.head_pitch_values.append(pitch)
        self.focus_scores.append(focus)
        if score < 60:
            self.distraction_frames.append(idx)
            LOGGER.debug("Frame %d: distraction detected. Score: %.2f | Reasons: %s", idx, score, reasons)

        for (x, y) in np.vstack([le, re]):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        scores         = np.array(self.attention_scores)
        avg_score      = float(np.mean(scores))
        grade          = ("A" if avg_score >= 75 else "B" if avg_score >= 60
                          else "C" if avg_score >= 50 else "D" if avg_score >= 40 else "F")
        verdict        = {
            "A": "Excellent attention", "B": "Good attention",
            "C": "Moderate attention",  "D": "Poor attention", "F": "Very poor"
        }[grade]
        total          = len(scores)
        focused_pct    = float(np.sum(scores >= 75)) / total * 100
        distracted_pct = float(np.sum(scores < 60))  / total * 100
        elapsed_sec    = max(1, (idx - self.start_sec * self.fps) // self.fps)
        blink_rate     = self.blink_count / elapsed_sec * 60

        return self._overlay(
            frame, score, gaze_h, gaze_v, ear, yaw, pitch, focus, reasons,
            avg_score, grade, verdict, focused_pct, distracted_pct, blink_rate
        )

    def analyze(self):
        end_frame   = self.end_sec   * self.fps
        start_frame = self.start_sec * self.fps
        frame_idx   = start_frame
        LOGGER.info("Attention analysis started. Frame range: %d → %d", start_frame, end_frame)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame_idx >= end_frame:
                break
            frame_idx += 1
            self.out.write(self._process_frame(frame, frame_idx))
            if frame_idx % self.fps == 0:
                elapsed = (frame_idx - start_frame) // self.fps
                avg     = np.mean(self.attention_scores) if self.attention_scores else 0
                bar     = "█" * int(avg // 5) + "░" * (20 - int(avg // 5))
                print(f"   {elapsed:>3}s | [{bar}] {avg:.1f}%")
                LOGGER.debug("Attention @ %ds: avg=%.1f%%", elapsed, avg)
        LOGGER.info("Attention frame loop complete. Total frames processed: %d", frame_idx - start_frame)

    def print_summary(self) -> str:
        if not self.attention_scores:
            LOGGER.warning("No attention data recorded.")
            return "  No attention data recorded."
        scores = np.array(self.attention_scores)
        avg    = float(np.mean(scores))
        std    = float(np.std(scores))
        total  = len(scores)
        focused    = int(np.sum(scores >= 75))
        moderate   = int(np.sum((scores >= 60) & (scores < 75)))
        distracted = int(np.sum(scores < 60))
        grade   = "A" if avg >= 75 else "B" if avg >= 60 else "C" if avg >= 50 else "D" if avg >= 40 else "F"
        verdict = {"A":"Excellent attention","B":"Good attention","C":"Moderate attention","D":"Poor attention","F":"Very poor"}[grade]
        avg_gh  = float(np.mean(self.gaze_h_values))   if self.gaze_h_values   else 0
        avg_gv  = float(np.mean(self.gaze_v_values))   if self.gaze_v_values   else 0
        avg_foc = float(np.mean(self.focus_scores))    if self.focus_scores     else 0
        avg_yaw = float(np.mean(np.abs(self.head_yaw_values)))   if self.head_yaw_values   else 0
        avg_pit = float(np.mean(np.abs(self.head_pitch_values))) if self.head_pitch_values else 0
        gs      = max(0.0, 100.0 - float(np.std(self.gaze_h_values)) * 300) if self.gaze_h_values else 0
        es      = max(0.0, 100.0 - float(np.mean(np.abs(np.diff(self.gaze_h_values)))) * 500) if len(self.gaze_h_values) > 1 else 100.0
        blink_rate = self.blink_count / max(1, self.end_sec - self.start_sec) * 60
        dist_secs  = sorted(set(f // self.fps for f in self.distraction_frames)) if self.distraction_frames else []
        bar_fill   = "█" * int(avg // 5)
        bar_empty  = "░" * (20 - int(avg // 5))
        lines = [
            "\n" + "═" * 58,
            "         ATTENTION ANALYSIS REPORT",
            "═" * 58,
            f"  Overall Score       : {avg:.1f}/100  [{grade}]",
            f"  Verdict             : {verdict}",
            f"  Std Dev             : ±{std:.1f}",
            f"  Min / Max           : {np.min(scores):.1f} / {np.max(scores):.1f}",
            "",
            f"  Focused   (≥75)     : {focused}/{total} frames ({focused/total*100:.1f}%)",
            f"  Moderate (60-74)    : {moderate}/{total} frames ({moderate/total*100:.1f}%)",
            f"  Distracted (<60)    : {distracted}/{total} frames ({distracted/total*100:.1f}%)",
            f"  No face frames      : {self.no_face_frames}",
            "",
            f"  Gaze H (centre=0.5) : {avg_gh:.3f}",
            f"  Gaze V (centre=0.5) : {avg_gv:.3f}",
            f"  Gaze Stability      : {gs:.1f}/100",
            f"  Eyeball Movement    : {es:.1f}/100",
            f"  Blinks              : {self.blink_count} ({blink_rate:.1f}/min, normal 15-20)",
            "",
            f"  Head Yaw            : {avg_yaw:.3f}",
            f"  Head Pitch          : {avg_pit:.3f}",
            f"  Focus Score         : {avg_foc:.1f}/100",
        ]
        if dist_secs:
            lines.append(f"  Distracted at (s)   : {dist_secs}")
        if "cell phone" in self.detected_objects:
            lines.append(f"  ⚠ Phone detected    : YES (−{self.PHONE_PENALTY:.0f}pts/frame penalty applied)")
        lines += [
            "",
            f"  [{bar_fill}{bar_empty}] {avg:.1f}%",
            f"  Output video        : {self.output_video}",
            "═" * 58,
        ]
        report = "\n".join(lines)
        print(report)
        METRICS_LOGGER.info(
            "Attention summary | score=%.1f grade=%s focused=%.1f%% distracted=%.1f%% "
            "blinks=%d blink_rate=%.1f/min gaze_stability=%.1f eyeball_movement=%.1f "
            "no_face_frames=%d",
            avg, grade, focused / total * 100, distracted / total * 100,
            self.blink_count, blink_rate, gs, es, self.no_face_frames
        )
        return report

    def release(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        LOGGER.debug("AttentionScoreAnalyzer resources released.")

    def run(self) -> str:
        try:
            self.setup()
            self.analyze()
            return self.print_summary()
        except Exception as exc:
            LOGGER.error("AttentionScoreAnalyzer.run() failed: %s\n%s", exc, traceback.format_exc())
            raise
        finally:
            self.release()


def analyze_attention_node(state: PipelineState) -> dict:
    LOGGER.info("NODE 3 started: Attention Score Analysis (dlib).")
    try:
        attention_end = settings.attention.end_sec
        if attention_end is not None and attention_end < 0:
            attention_end = None  # -1 means full video
        analyzer = AttentionScoreAnalyzer(
            video_path=state["video_path"],
            start_sec=settings.attention.start_sec,
            end_sec=attention_end,
            detected_objects=state.get("detected_objects", {}),
        )
        report = analyzer.run()
        LOGGER.info("NODE 3 complete. Output video: %s", analyzer.output_video)
        return {
            "attention_report":       report,
            "attention_output_video": analyzer.output_video,
        }
    except Exception as exc:
        LOGGER.error("NODE 3 failed: %s\n%s", exc, traceback.format_exc())
        raise


# ─── Node 4 — LLM Data Extraction ────────────────────────────────────────────

def _build_extraction_prompt(state: PipelineState) -> str:
    schema_lines = "\n".join(
        f'  "{k}": "{v}"'
        for k, v in EXTRACTION_SCHEMA.items()
    )
    transcript       = state.get("transcript", "").strip() or "N/A"
    attention_report = state.get("attention_report", "").strip() or "N/A"
    face_detected    = str(state.get("face_detected", False))
    detected_objects = json.dumps(state.get("detected_objects", {}))
    attention_video  = state.get("attention_output_video", "").strip() or "N/A"
    lang_name        = state.get("language_name", "English")
    lang_code        = state.get("language_code", "en")
    transcript_note = ""
    if lang_code != "en":
        transcript_note = (
            f"\n[NOTE] The video language is '{lang_name}' ({lang_code}). "
            "Transcription model is not yet available for this language, "
            "so the transcript is unavailable. Extract all non-transcript fields "
            "from the attention/face data provided.\n"
        )

    LOGGER.debug(
        "Building LLM extraction prompt. Language: %s | Transcript length: %d",
        lang_name, len(transcript)
    )

    # ── CHANGE 2: Expanded and more explicit name-extraction instructions ──
    return f"""You are an intelligent data-extraction assistant.
Below is the combined output of a video analysis pipeline that processed an insurance consent recording.
Extract every field listed in the schema below.
- Return ONLY a single valid JSON object.  No markdown, no explanation, no preamble.
- If a field cannot be determined from the provided data, set its value to "N/A".
- Do not hallucinate values not present in the data.
- If there is a consent statement, then data_consent_to_insurer is also Yes and consent_given is also Yes.

CUSTOMER NAME EXTRACTION — follow these steps in order:
  1. Search the consent statement for a name the speaker introduces themselves with
     (e.g. "I, <Name>, hereby...", "My name is <Name>", "This is <Name> speaking").
  2. If not found there, scan the entire transcript for any self-introduction or
     spoken name reference (e.g. "मेरा नाम <Name> है", "என் பெயர் <Name>",
     "I am <Name>", "My name is <Name>").
  3. Look for the name on any on-screen text, captions, or spoken identification
     that appears in the transcript data.
  4. Only after exhausting all of the above, set customer_name to "N/A".
{transcript_note}
━━━ EXTRACTION SCHEMA ━━━
{{
{schema_lines}
}}

━━━ PIPELINE OUTPUT ━━━

[LANGUAGE]
language_name : {lang_name}
language_code : {lang_code}

[TRANSCRIPT]
{transcript}

[FACE DETECTION]
face_detected     : {face_detected}
detected_objects  : {detected_objects}

[ATTENTION ANALYSIS REPORT]
{attention_report}

[OUTPUT VIDEO]
attention_output_video : {attention_video}

━━━ OUTPUT (JSON only) ━━━"""


def _call_llm(prompt: str) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise data-extraction assistant. "
                    "Always respond with a single valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    LOGGER.info("Calling LLM API. URL: %s | Model: %s", LLM_API_URL, LLM_MODEL)

    response = None
    try:
        response = requests.post(
            LLM_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        raw     = response.json()
        content = raw["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        LOGGER.info("LLM extraction successful. Fields returned: %d", len(parsed))
        return parsed

    except requests.exceptions.RequestException as exc:
        LOGGER.error("LLM HTTP error: %s\n%s", exc, traceback.format_exc())
        return {k: "N/A" for k in EXTRACTION_SCHEMA}
    except (json.JSONDecodeError, KeyError) as exc:
        raw_text = response.text[:500] if response is not None else "<no response body>"
        LOGGER.error(
            "LLM response parse error: %s | Raw response (first 500 chars): %s\n%s",
            exc, raw_text, traceback.format_exc()
        )
        return {k: "N/A" for k in EXTRACTION_SCHEMA}


def _fill_missing_with_na(data: dict) -> dict:
    result = {}
    for key in EXTRACTION_SCHEMA:
        val = data.get(key, "N/A")
        if val is None or (isinstance(val, str) and val.strip() == ""):
            val = "N/A"
        result[key] = val
    return result


def _format_extraction_report(data: dict) -> str:
    lines = [
        "\n" + "═" * 58,
        "         LLM EXTRACTED DATA REPORT",
        "═" * 58,
    ]
    max_key_len = max(len(k) for k in data)
    for key, val in data.items():
        lines.append(f"  {key:<{max_key_len}} : {val}")
    lines.append("═" * 58)
    return "\n".join(lines)


def extract_data_node(state: PipelineState) -> dict:
    LOGGER.info("NODE 4 started: LLM Data Extraction.")
    try:
        prompt    = _build_extraction_prompt(state)
        raw_data  = _call_llm(prompt)
        extracted = _fill_missing_with_na(raw_data)
        report    = _format_extraction_report(extracted)
        print(report)

        output_json_path = "extracted_data.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        LOGGER.info("Extracted data saved to: %s", output_json_path)

        LOGGER.info("NODE 4 complete.")
        return {
            "extracted_data":    extracted,
            "extraction_report": report,
        }
    except Exception as exc:
        LOGGER.error("NODE 4 failed: %s\n%s", exc, traceback.format_exc())
        raise


# ─── Graph / Pipeline ────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    LOGGER.info("Building LangGraph pipeline.")
    graph = StateGraph(PipelineState)
    graph.add_node("transcribe_audio",  transcribe_audio_node)
    graph.add_node("detect_faces",      detect_faces_node)
    graph.add_node("analyze_attention", analyze_attention_node)
    graph.add_node("extract_data",      extract_data_node)
    graph.add_edge(START,               "transcribe_audio")
    graph.add_edge("transcribe_audio",  "detect_faces")
    graph.add_conditional_edges(
        "detect_faces",
        route_after_face_detection,
        {
            "analyze_attention": "analyze_attention",
            "extract_data":      "extract_data",
        },
    )
    graph.add_edge("analyze_attention", "extract_data")
    graph.add_edge("extract_data",      END)
    LOGGER.info("Pipeline graph built successfully.")
    return graph


def run_pipeline(video_path: str, language_name: str, language_code: str) -> PipelineState:
    LOGGER.info(
        "Pipeline started. Video: %s | Language: %s (%s)",
        video_path, language_name, language_code
    )

    uncompiled_pipeline = build_pipeline()
    pipeline = uncompiled_pipeline.compile()

    initial_state: PipelineState = {
        "video_path":             video_path,
        "language_name":          language_name,
        "language_code":          language_code,
        "transcript":             "",
        "face_detected":          False,
        "detected_objects":       {},
        "face_image_path":        "",
        "yolo_output_video":      "",
        "face_detection_report":  "",
        "attention_report":       "",
        "attention_output_video": "",
        "extracted_data":         {},
        "extraction_report":      "",
    }

    try:
        final_state = pipeline.invoke(initial_state)
    except Exception as exc:
        LOGGER.critical("Pipeline invocation failed: %s\n%s", exc, traceback.format_exc())
        raise

    # ── CHANGE 3: Print full transcript again in the pipeline summary (for
    #             non-English runs where NODE 1 printed nothing, and as a
    #             final consolidated summary section) ──
    separator = "═" * 58
    print(f"\n{separator}")
    print("         PIPELINE SUMMARY")
    print(separator)

    if language_code == "en":
        transcript_text = final_state.get("transcript", "").strip()
        if transcript_text:
            # Re-print for convenience at the end of the pipeline summary
            print(f"\n  FULL TRANSCRIPT [{language_name}]:")
            print(f"  {'─' * 54}")
            # Wrap long transcript with indentation for readability
            for line in transcript_text.splitlines():
                print(f"  {line}")
            print(f"  {'─' * 54}")
        else:
            print("  TRANSCRIPT: (empty — no speech detected)")
        LOGGER.info("FULL TRANSCRIPT:\n%s", final_state.get("transcript", ""))
    else:
        print(f"  TRANSCRIPT: SKIPPED — {language_name} model not available.")
        LOGGER.info("TRANSCRIPT SKIPPED — %s model not available.", language_name)

    print(f"\n  Face detected   : {final_state['face_detected']}")
    print(f"  YOLO video      : {final_state['yolo_output_video']}")

    if final_state["face_detected"]:
        print(f"  Attention video : {final_state['attention_output_video']}")
        LOGGER.info("Attention video : %s", final_state["attention_output_video"])
        LOGGER.info("FULL ATTENTION REPORT:\n%s", final_state["attention_report"])

    # Customer name quick-view
    customer_name = final_state.get("extracted_data", {}).get("customer_name", "N/A")
    print(f"\n  Customer name   : {customer_name}")
    print(separator)

    LOGGER.info("EXTRACTED DATA:\n%s", final_state["extraction_report"])
    LOGGER.info(
        "Pipeline complete. Face detected: %s | Extracted fields: %d",
        final_state["face_detected"], len(final_state["extracted_data"])
    )
    METRICS_LOGGER.info(
        "Pipeline metrics | video=%s language=%s face_detected=%s extracted_fields=%d customer_name=%s",
        video_path, language_code, final_state["face_detected"],
        len(final_state["extracted_data"]), customer_name
    )
    return final_state


# ─── Helpers ─────────────────────────────────────────────────────────────────

def download_dlib_model():
    model_path = "shape_predictor_68_face_landmarks.dat"
    bz2_path   = "shape_predictor_68_face_landmarks.dat.bz2"
    model_url  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    if os.path.exists(model_path):
        LOGGER.info("dlib model already present: %s", model_path)
        return

    if not os.path.exists(bz2_path):
        LOGGER.info("Downloading dlib model from %s", model_url)
        subprocess.run(["wget", "-q", "--show-progress", model_url], check=True)
        LOGGER.info("dlib model download complete.")
    else:
        LOGGER.info("dlib archive already downloaded: %s", bz2_path)

    LOGGER.info("Extracting dlib model archive.")
    subprocess.run(["bzip2", "-d", bz2_path], check=True)
    LOGGER.info("dlib model ready: %s", model_path)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LOGGER.info("Application started.")
    try:
        download_dlib_model()
        language_name, language_code = select_language()
        video_file_path = (
            '/home/muralik/Downloads/18-Mohammed Ayan-customer is not concentrating on BOT audio.mp4'
        )
        run_pipeline(video_file_path, language_name, language_code)
        LOGGER.info("Application finished successfully.")
    except Exception as exc:
        LOGGER.critical("Unhandled exception at top level: %s\n%s", exc, traceback.format_exc())
        raise