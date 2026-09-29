"""Tweakable settings for the Call and Response prototype."""
import os

# --- LLM ("NLI") ---------------------------------------------------------
# "auto" = Claude if ANTHROPIC_API_KEY is set, else Ollama if it's running, else offline.
LLM_BACKEND = os.getenv("RHYTHM_LLM_BACKEND", "auto")   # auto | anthropic | ollama | offline

# Free local option: install Ollama, then `ollama pull llama3.1:8b`
OLLAMA_MODEL = os.getenv("RHYTHM_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Paid option: set ANTHROPIC_API_KEY in your environment
LLM_MODEL = os.getenv("RHYTHM_LLM_MODEL", "claude-sonnet-5-5")

# --- Speech-to-text ------------------------------------------------------
# faster-whisper model: "tiny.en" (fastest), "base.en" (good default), "small.en" (most accurate)
WHISPER_MODEL = os.getenv("RHYTHM_WHISPER_MODEL", "base.en")
SAMPLE_RATE = 16000

# --- Beat ----------------------------------------------------------------
MIN_BPM = 60
MAX_BPM = 110
BPM_STEP = 6
BEAT_VOLUME = 0.25          # keep low so the mic doesn't pick it up much (headphones help)
BEATS_PER_BAR = 4

# --- Camera ---------------------------------------------------------------
# "pepper" = Pepper's head camera (run pepper_camera_server.py on the robot first)
# "webcam" = laptop webcam (handy for testing without the robot)
CAMERA_SOURCE = os.getenv("RHYTHM_CAMERA", "pepper")
PEPPER_IP = os.getenv("PEPPER_IP", "192.168.1.100")    # <- change to your Pepper's IP
PEPPER_CAMERA_PORT = 5566

# --- Gestures ------------------------------------------------------------
SHOW_ACTION_DURING_TURN = True   # keep the robot doing the move on screen while the child copies it
                                 # (False = child has to remember it)

# --- Game ----------------------------------------------------------------
SCREENING_PHRASE_COUNT = 6
PHRASES_PER_LEVEL = 5
MAX_ATTEMPTS = 2            # first try + one gentle retry
EXTRA_RESPONSE_BEATS = 6    # response window = (words in phrase + this) beats
MIN_RESPONSE_SECONDS = 4.0

# --- Scoring thresholds ---------------------------------------------------
CORRECT_PHONETIC = 0.88     # phonetic score needed (and no missing words) for "correct"
PARTIAL_PHONETIC = 0.45     # below this (and low semantic score) -> "incorrect"
PARTIAL_SEMANTIC = 0.70

# Whisper confidence below this marks a word as "unclear" (likely mispronounced), even if
# the transcript shows the right word. Raise (e.g. 0.7) = stricter; lower (e.g. 0.35) = more forgiving.
UNCLEAR_WORD_CONFIDENCE = 0.55
SHOW_WORD_CONFIDENCE = True   # print each word's confidence so you can tune the number above

POINTS = {"correct_first": 10, "correct_retry": 7, "partial": 4, "incorrect": 1, "no_response": 0}