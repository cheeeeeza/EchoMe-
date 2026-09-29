"""Speech-to-text (faster-whisper, runs locally) + response-latency detection."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import config

# Hugging Face caches the Whisper model in ~/.cache/huggingface by default. Where that
# folder is not writable (restricted shells, Defender-controlled folders) the very first
# run dies with PermissionError before the game can start, so keep the cache beside this
# file instead. An explicitly set HF_HOME still takes precedence.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf"))


@dataclass
class Heard:
    transcript: str
    onset_s: float | None   # seconds from "your turn" until the child started talking
    words: list[tuple[str, float]] | None = None   # (word, Whisper confidence 0-1)


class Transcriber:
    def __init__(self):
        from faster_whisper import WhisperModel

        print(f"Loading speech model ({config.WHISPER_MODEL})... first run downloads it.")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.noise_floor = 0.01

    def calibrate(self, silence: np.ndarray):
        """Measure room noise so onset detection isn't fooled by fans, the beat, etc."""
        rms = _frame_rms(silence)
        if len(rms):
            self.noise_floor = float(np.percentile(rms, 90))

    def transcribe(self, audio: np.ndarray) -> Heard:
        onset = detect_onset(audio, self.noise_floor)
        if onset is None:
            return Heard("", None)
        # NOTE: deliberately NO initial_prompt with the target phrase — that would bias
        # Whisper toward "hearing" the correct words and hide real mispronunciations.
        # word_timestamps=True gives a confidence score per word. When Whisper "autocorrects"
        # a mispronounced word into the real word, its confidence in that word usually drops.
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True,
                                            beam_size=5, condition_on_previous_text=False,
                                            word_timestamps=True, temperature=0.0)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        words = [(w.word.strip(), float(w.probability)) for s in segments for w in (s.words or [])]
        return Heard(text, onset if text else None, words)


def _frame_rms(audio: np.ndarray, frame_ms: int = 30) -> np.ndarray:
    n = int(config.SAMPLE_RATE * frame_ms / 1000)
    frames = len(audio) // n
    if frames == 0:
        return np.array([])
    return np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1))


def detect_onset(audio: np.ndarray, noise_floor: float, frame_ms: int = 30) -> float | None:
    """First time speech energy stays above the noise floor for ~150 ms."""
    rms = _frame_rms(audio, frame_ms)
    threshold = max(0.015, noise_floor * 3)
    loud = rms > threshold
    need = 5  # consecutive frames
    for i in range(len(loud) - need):
        if loud[i:i + need].all():
            return i * frame_ms / 1000
    return None