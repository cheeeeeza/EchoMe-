"""Encouraging feedback lines + adaptive difficulty."""
from __future__ import annotations

import random

import config
from matching import MatchResult

FEEDBACK = {
    "correct": ["Yes! You got it!", "Awesome, right on the beat!", "Perfect! High five!",
                "Wow, you nailed that one!", "Great job, that sounded super clear!"],
    "correct_retry": ["There it is! Great job trying again.", "Yes! You kept going and got it!",
                      "Brilliant, that one was tricky and you did it!"],
    "partial": ["Ooh, so close! Let's try that one more time.", "Nice try! Listen once more.",
                "Good effort! Let's say it together again."],
    "incorrect": ["That was a good try. Let's listen again.", "Thanks for having a go! Here it is again."],
    "move_on": ["Good trying! Let's do a new one.", "Nice work sticking with it. Next one!"],
    "no_response": ["That's okay. We can take a little break.",
                    "No rush. Take a nice breath."],
    "no_response_ask": ["Do you want to try again, or skip this one? You can say try again, or skip."],
    "skip": ["Okay, let's skip that one. Here comes a new one!"],
    "up_longer": ["You're doing great, let's try a longer one!", "Wow! Ready for a bigger one?"],
    "up_faster": ["You're on fire! Let's make the beat a little faster.", "Let's speed it up a tiny bit!"],
    "down": ["Let's slow it down and try a shorter one together.", "Let's make it a bit easier and slower."],
}


def line(kind: str) -> str:
    return random.choice(FEEDBACK[kind])


def hint(result: MatchResult) -> str | None:
    """One gentle, specific pointer — never a list of everything wrong."""
    if result.missing:
        return f"Listen for the word '{result.missing[0]}'."
    if result.mispronounced:
        return f"Listen to how I say '{result.mispronounced[0][0]}'."
    return None


class Difficulty:
    """Tracks bpm and phrase-length tier; steps up after success, down after struggle."""

    def __init__(self, bpm: int, tier: int):
        self.bpm = int(max(config.MIN_BPM, min(config.MAX_BPM, bpm)))
        self.tier = max(0, min(2, tier))
        self.history: list[bool] = []

    def update(self, success: bool) -> str | None:
        self.history.append(success)
        last = self.history[-2:]
        if len(last) == 2 and all(last):
            self.history.clear()
            if self.tier < 2:
                self.tier += 1
                return "up_longer"
            if self.bpm < config.MAX_BPM:
                self.bpm = min(config.MAX_BPM, self.bpm + config.BPM_STEP)
                return "up_faster"
            return None
        if len(last) == 2 and not any(last):
            self.history.clear()
            if self.bpm > config.MIN_BPM + config.BPM_STEP:
                self.bpm -= config.BPM_STEP
            if self.tier > 0:
                self.tier -= 1
            return "down"
        return None
