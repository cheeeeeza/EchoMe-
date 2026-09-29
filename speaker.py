"""Room-noise calibration: how loud is the room, and how loud should Pepper be?

Runs once before the game starts (see game.Session.calibrate). The laptop microphone
records the room while everyone stays quiet, measures how loud it is, and maps that
onto an output level: a noisy room gets a louder Pepper, a quiet room a quieter one.
"""
from __future__ import annotations

import numpy as np

import config
from speech import _frame_rms   # same framing the onset detector uses


def room_level_db(audio: np.ndarray) -> float:
    """Room tone in dBFS (0 dB = digital full scale, so quieter rooms go more negative)."""
    rms = _frame_rms(audio)
    if not len(rms):
        return config.CALIBRATE_LOW_DB
    # 90th percentile: shrug off a single cough or chair scrape, keep the steady background
    level = float(np.percentile(rms, 90))
    return float(20.0 * np.log10(max(level, 1e-9)))


def volume_for_db(level_db: float) -> float:
    """Map room loudness onto a speaker volume (0-1), linear in dB so it sounds even."""
    lo, hi = config.CALIBRATE_LOW_DB, config.CALIBRATE_HIGH_DB
    frac = (level_db - lo) / max(hi - lo, 1e-6)
    frac = float(np.clip(frac, 0.0, 1.0))
    return config.SPEAKER_VOLUME_MIN + frac * (config.SPEAKER_VOLUME_MAX - config.SPEAKER_VOLUME_MIN)


def calibrate(audio: np.ndarray) -> tuple[float, float]:
    """Return (room level in dBFS, suggested speaker volume 0-1) for this recording."""
    level_db = room_level_db(audio)
    return level_db, volume_for_db(level_db)
