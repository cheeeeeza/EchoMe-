"""
Gestures the robot can pair with a phrase.

Each action has:
  meaning  - told to the LLM so it only picks actions that fit the phrase
  cue      - what the robot says if the child forgets the move
  demo     - stick-figure keyframes for the on-screen robot (laptop stand-in for Pepper)
  detector - checks MediaPipe pose landmarks to see if the child did it

Landmarks use MediaPipe Pose indices, coordinates normalised 0-1 (y grows downward).
"""
from __future__ import annotations

import math
from collections import deque

# ------------------------------------------------------------------ vocabulary
ACTIONS = {
    "wave":          {"meaning": "wave one hand: hello, goodbye, hi, see you",
                      "cue": "wave your hand"},
    "arms_up":       {"meaning": "both arms up high: happy, excited, hooray, yay, I did it",
                      "cue": "put your arms up high"},
    "hug_self":      {"meaning": "cross arms and hug yourself: calm, safe, love, comfort, gentle",
                      "cue": "give yourself a hug"},
    "hand_on_heart": {"meaning": "hand on chest: thank you, sorry, I care, feelings, kind",
                      "cue": "put your hand on your heart"},
    "hands_on_head": {"meaning": "both hands on head: thinking, oops, confused, I forgot",
                      "cue": "put your hands on your head"},
    "clap":          {"meaning": "clap hands: well done, good job, celebrating, let's play",
                      "cue": "clap your hands"},
    "arms_out":      {"meaning": "stretch arms out wide: welcome, everyone, together, big, share",
                      "cue": "stretch your arms out wide"},
}


def valid(name) -> str | None:
    """Clean up whatever the LLM returned into a known action name or None."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    return name if name in ACTIONS else None


def describe_for_llm() -> str:
    return "\n".join(f'- "{k}": {v["meaning"]}' for k, v in ACTIONS.items())


# ------------------------------------------------------------------ robot demo figure
# Viewer coordinates in a 260x300 panel. Each keyframe: (L_elbow, L_wrist, R_elbow, R_wrist).
_IDLE_L, _IDLE_R = ((90, 150), (88, 190)), ((170, 150), (172, 190))
DEMO_KEYFRAMES = {
    None:            [(*_IDLE_L, *_IDLE_R)],
    "wave":          [(*_IDLE_L, (185, 95), (190, 45)), (*_IDLE_L, (185, 95), (220, 55))],
    "arms_up":       [((90, 70), (95, 25), (170, 70), (165, 25))],
    "hug_self":      [((105, 150), (152, 118), (155, 150), (108, 118))],
    "hand_on_heart": [(*_IDLE_L, (160, 150), (118, 128))],
    "hands_on_head": [((78, 78), (115, 52), (182, 78), (145, 52))],
    "clap":          [((105, 140), (124, 112), (155, 140), (136, 112)),
                      ((100, 140), (98, 110), (160, 140), (162, 110))],
    "arms_out":      [((65, 110), (25, 110), (195, 110), (235, 110))],
}


def demo_frame(action: str | None, t: float):
    frames = DEMO_KEYFRAMES.get(action, DEMO_KEYFRAMES[None])
    return frames[int(t * 3) % len(frames)]   # animated actions flip ~3x per second


# ------------------------------------------------------------------ detectors
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
_NEEDED = (NOSE, L_SH, R_SH, L_WR, R_WR)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Detector:
    """Feed it pose frames; `done` flips True once the action is seen."""
    HOLD_S = 0.35   # static poses must be held this long

    def __init__(self, action: str):
        self.action = action
        self.done = False
        self.done_at: float | None = None
        self._since: float | None = None
        self._hist: deque = deque(maxlen=45)   # ~1.5 s of frames for movement actions

    def update(self, lm, t: float) -> bool:
        """lm: list of (x, y, visibility) for 33 landmarks, or None if no person."""
        if self.done:
            return True
        if lm is None or any(lm[i][2] < 0.4 for i in _NEEDED):
            self._since = None
            return False
        P = {i: (lm[i][0], lm[i][1]) for i in range(len(lm))}
        sw = max(_d(P[L_SH], P[R_SH]), 1e-3)          # shoulder width = body scale
        hit = getattr(self, "_" + self.action)(P, sw, t)
        if hit is True:                                 # static pose -> must be held
            self._since = self._since or t
            if t - self._since >= self.HOLD_S:
                self._finish(t)
        elif hit == "now":                              # movement pattern completed
            self._finish(t)
        else:
            self._since = None
        return self.done

    def _finish(self, t):
        self.done, self.done_at = True, t

    # --- static poses ---
    def _arms_up(self, P, sw, t):
        return P[L_WR][1] < P[NOSE][1] - 0.8 * sw and P[R_WR][1] < P[NOSE][1] - 0.8 * sw

    def _hands_on_head(self, P, sw, t):
        ok = lambda w: (P[NOSE][1] - 0.8 * sw < P[w][1] < P[L_SH][1] - 0.1 * sw
                        and abs(P[w][0] - P[NOSE][0]) < 0.9 * sw)
        return ok(L_WR) and ok(R_WR)

    def _hug_self(self, P, sw, t):
        return _d(P[L_WR], P[R_SH]) < 0.55 * sw and _d(P[R_WR], P[L_SH]) < 0.55 * sw

    def _hand_on_heart(self, P, sw, t):
        chest = ((P[L_SH][0] + P[R_SH][0]) / 2, (P[L_SH][1] + P[R_SH][1]) / 2 + 0.35 * sw)
        near = min(_d(P[L_WR], chest), _d(P[R_WR], chest)) < 0.45 * sw
        return near and not self._hug_self(P, sw, t)   # one or both hands, but not a hug

    def _arms_out(self, P, sw, t):
        level = lambda w, s: abs(P[w][1] - P[s][1]) < 0.45 * sw
        return level(L_WR, L_SH) and level(R_WR, R_SH) and _d(P[L_WR], P[R_WR]) > 2.4 * sw

    # --- movement patterns ---
    def _wave(self, P, sw, t):
        # a raised hand (above shoulder) swinging side to side
        raised = [w for w, s in ((L_WR, L_SH), (R_WR, R_SH)) if P[w][1] < P[s][1]]
        if not raised:
            self._hist.clear()
            return False
        self._hist.append(P[raised[0]][0] / sw)
        return "now" if _direction_changes(list(self._hist), min_swing=0.18) >= 2 else False

    def _clap(self, P, sw, t):
        self._hist.append(_d(P[L_WR], P[R_WR]) / sw)
        closes, was_open = 0, False
        for g in self._hist:
            if g > 0.8:
                was_open = True
            elif g < 0.35 and was_open:
                closes, was_open = closes + 1, False
        return "now" if closes >= 2 else False


def _direction_changes(xs, min_swing):
    changes, direction, anchor = 0, 0, xs[0] if xs else 0
    for x in xs[1:]:
        if direction >= 0 and x < anchor - min_swing:
            changes += direction != 0
            direction, anchor = -1, x
        elif direction <= 0 and x > anchor + min_swing:
            changes += direction != 0
            direction, anchor = 1, x
        elif (direction > 0 and x > anchor) or (direction < 0 and x < anchor):
            anchor = x
    return changes