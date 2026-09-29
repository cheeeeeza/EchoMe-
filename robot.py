"""
Robot abstraction.

The game only talks to the `Robot` interface, so swapping your laptop for
Pepper later means writing one new class (see PepperRobot at the bottom)
without touching the game logic.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

import config


class Robot(ABC):
    @abstractmethod
    def say(self, text: str, bpm: int | None = None) -> None:
        """Speak text. If bpm is given, speak at a pace that fits the beat."""

    @abstractmethod
    def start_beat(self, bpm: int) -> None: ...

    @abstractmethod
    def stop_beat(self) -> None: ...

    @abstractmethod
    def set_bpm(self, bpm: int) -> None: ...

    @abstractmethod
    def wait_for_downbeat(self) -> None:
        """Block until the start of the next bar so speech lands on beat 1."""

    @abstractmethod
    def cue_turn(self) -> None:
        """Play the 'your turn' sound."""

    @abstractmethod
    def record(self, seconds: float) -> np.ndarray:
        """Record mono float32 audio at config.SAMPLE_RATE."""

    @abstractmethod
    def do_action(self, action: str | None) -> None:
        """Start a gesture from actions.ACTIONS (None = back to neutral). Non-blocking."""

    def start_recording(self, seconds: float) -> None:
        """Start recording without blocking (so the camera can run at the same time)."""
        raise NotImplementedError

    def finish_recording(self) -> np.ndarray:
        raise NotImplementedError

    def express(self, mood: str) -> None:
        """Hook for gestures/LEDs on a real robot (e.g. 'happy', 'encourage', 'calm')."""
        pass


# ---------------------------------------------------------------------------
# Laptop implementation
# ---------------------------------------------------------------------------
class _Metronome:
    """Click track generated in a sounddevice callback, so it never drifts."""

    def __init__(self, sample_rate: int = 44100):
        import sounddevice as sd

        self.sd = sd
        self.sr = sample_rate
        self.bpm = 80
        self.stream = None
        self._pos = 0                    # samples since start
        self._lock = threading.Lock()
        self._click_hi = self._tone(1500, 0.03)
        self._click_lo = self._tone(1000, 0.03)
        self._chime = np.concatenate([self._tone(880, 0.12), self._tone(1320, 0.18)])
        self._pending: list[np.ndarray] = []   # one-shot sounds to mix in
        self._pending_offset = 0
        self._start_time = 0.0

    def _tone(self, freq, dur):
        t = np.arange(int(self.sr * dur)) / self.sr
        env = np.exp(-t * 40 / dur / 10)
        return (np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

    @property
    def spb(self):  # samples per beat
        return int(self.sr * 60 / self.bpm)

    def _callback(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        with self._lock:
            spb = self.spb
            idx = self._pos + np.arange(frames)
            beat_pos = idx % spb
            accent = (idx // spb) % config.BEATS_PER_BAR == 0
            mask = beat_pos < len(self._click_hi)
            bp = beat_pos[mask]
            buf[mask] = np.where(accent[mask], self._click_hi[bp], self._click_lo[bp]) * config.BEAT_VOLUME
            if self._pending:
                snd = self._pending[0]
                chunk = snd[self._pending_offset:self._pending_offset + frames]
                buf[:len(chunk)] += chunk * 0.5
                self._pending_offset += frames
                if self._pending_offset >= len(snd):
                    self._pending.pop(0)
                    self._pending_offset = 0
            self._pos += frames
        outdata[:, 0] = buf

    def start(self, bpm):
        self.bpm = bpm
        if self.stream is None:
            self._pos = 0
            self._start_time = time.monotonic()
            self.stream = self.sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                               callback=self._callback)
            self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_bpm(self, bpm):
        with self._lock:
            # keep phase: convert position to beats, rescale
            beats = self._pos / self.spb
            self.bpm = bpm
            self._pos = int(beats * self.spb)

    def seconds_to_next_bar(self):
        with self._lock:
            spbar = self.spb * config.BEATS_PER_BAR
            remaining = spbar - (self._pos % spbar)
        return remaining / self.sr

    def chime(self):
        with self._lock:
            self._pending.append(self._chime)


class LocalRobot(Robot):
    """Uses your laptop speakers + microphone. `speak=False` prints instead of talking."""

    def __init__(self, speak: bool = True, beat: bool = True):
        self.speak = speak
        self.beat_enabled = beat
        self._metro = None
        self._beat_on = False

    def _metronome(self):
        if self._metro is None:
            self._metro = _Metronome()
        return self._metro

    # --- speech ---
    def say(self, text, bpm=None):
        print(f"🤖 Robot: {text}")
        if not self.speak:
            return
        # Roughly one word per beat-and-a-half: slow, clear, rhythmic
        wpm = int(np.clip((bpm or 100) * 1.5, 110, 175))
        if sys.platform == "darwin" and shutil.which("say"):
            subprocess.run(["say", "-r", str(wpm), text])
        else:
            import pyttsx3  # re-init each call: avoids pyttsx3's "hangs on 2nd call" bug
            engine = pyttsx3.init()
            engine.setProperty("rate", wpm)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    # --- beat ---
    def start_beat(self, bpm):
        if not self.beat_enabled:
            return
        self._metronome().start(bpm)
        self._beat_on = True

    def stop_beat(self):
        if self._metro:
            self._metro.stop()
        self._beat_on = False

    def set_bpm(self, bpm):
        if self._metro:
            self._metro.set_bpm(bpm)

    def wait_for_downbeat(self):
        if self._beat_on:
            time.sleep(self._metro.seconds_to_next_bar())

    def cue_turn(self):
        if self._beat_on:
            self._metro.chime()
            time.sleep(0.3)
        print("🎤 Your turn!")

    # --- gestures ---
    def do_action(self, action):
        # The laptop has no arms: vision.py draws the robot doing it on screen.
        if action:
            print(f"   🤖 does: {action.replace('_', ' ')}")

    # --- mic ---
    def start_recording(self, seconds):
        import sounddevice as sd
        self._rec = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE,
                           channels=1, dtype="float32")

    def finish_recording(self):
        import sounddevice as sd
        sd.wait()
        return self._rec[:, 0]

    def record(self, seconds):
        import sounddevice as sd

        audio = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE,
                       channels=1, dtype="float32")
        # simple visual countdown
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            left = end - time.monotonic()
            bar = "█" * int(left) + " " * int(seconds - left)
            print(f"\r   listening {bar} {left:4.1f}s ", end="", flush=True)
            time.sleep(0.1)
        sd.wait()
        print("\r" + " " * 60 + "\r", end="")
        return audio[:, 0]


# ---------------------------------------------------------------------------
# Laptop mic + Pepper's voice
# ---------------------------------------------------------------------------
class PepperSpeakerRobot(LocalRobot):
    """
    Pepper speaks, keeps the beat and plays the "your turn" chime; the laptop
    microphone records the child. Needs pepper_server.py running on the robot.
    The beat runs on Pepper itself, so Wi-Fi delays can't make it wobble.
    """

    def __init__(self, ip: str, port: int = 5567, beat: bool = True):
        super().__init__(speak=True, beat=beat)
        self._lock = threading.Lock()
        try:
            self._sock = socket.create_connection((ip, port), timeout=5)
        except OSError as e:
            raise RuntimeError(f"Couldn't reach Pepper's speech/beat server at {ip}:{port} ({e}). "
                               "Is pepper_server.py running on the robot?") from e
        self._sock.settimeout(None)
        self._reader = self._sock.makefile("r", encoding="utf-8")
        self._downbeat_next = False
        self._send({"op": "ping"})
        print(f"🔊 Pepper is the speaker ({ip})")

    def _send(self, cmd: dict) -> dict:
        with self._lock:
            self._sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
            line = self._reader.readline()
        if not line:
            raise ConnectionError("Lost connection to Pepper")
        reply = json.loads(line)
        if not reply.get("ok"):
            print(f"   ⚠️  Pepper: {reply.get('error')}")
        return reply

    def say(self, text, bpm=None):
        print(f"🤖 Robot: {text}")
        # Same pacing idea as the laptop voice, as a Pepper speed percentage (100 = normal)
        wpm = float(np.clip((bpm or 100) * 1.5, 110, 175))
        speed = int(np.clip(wpm / 170 * 100, 75, 110))   # Pepper speed %; tune if it sounds too slow/fast
        self._send({"op": "say", "text": text, "speed": speed, "downbeat": self._downbeat_next})
        self._downbeat_next = False

    def start_beat(self, bpm):
        if self.beat_enabled:
            self._send({"op": "beat_start", "bpm": int(bpm)})
            self._beat_on = True

    def stop_beat(self):
        if self._beat_on:
            self._send({"op": "beat_stop"})
        self._beat_on = False

    def set_bpm(self, bpm):
        if self._beat_on:
            self._send({"op": "set_bpm", "bpm": int(bpm)})

    PEPPER_GESTURES = {"wave"}        # the rest still show on the laptop's stick figure

    def do_action(self, action):
        super().do_action(action)
        if action in self.PEPPER_GESTURES:
            self._send({"op": "gesture", "name": action})

    def wait_for_downbeat(self):
        # Pepper waits for the bar itself, so the timing stays tight over Wi-Fi
        self._downbeat_next = self._beat_on

    def cue_turn(self):
        self._send({"op": "chime"})
        print("🎤 Your turn!")

    def close(self):
        try:
            self.stop_beat()
            self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Pepper (future)
# ---------------------------------------------------------------------------
class PepperRobot(Robot):
    """
    Sketch of the Pepper version. Needs the `qi` SDK and a connection like:
        session = qi.Session(); session.connect("tcp://<pepper-ip>:9559")
    Mapping:
        say           -> ALAnimatedSpeech.say(text)  (speed via \\rspd=NN\\ tag)
        beat / cue    -> ALAudioPlayer.playFile / playSine on the robot
        record        -> ALAudioDevice.subscribe + processRemote (stream mic to laptop),
                         then run Whisper on the laptop as we do now
        express       -> ALLeds / ALAnimationPlayer ('happy', 'calm' animations)
        do_action     -> ALAnimationPlayer.run(<animation path>, _async=True) or
                         ALMotion.angleInterpolation on the arm joints, one entry per
                         action in actions.ACTIONS (wave, arms_up, hug_self, ...)
        camera        -> keep the laptop webcam, or read Pepper's head camera via ALVideoDevice
                         and feed the frames into vision.py
    """

    def __init__(self, ip: str, port: int = 9559):
        raise NotImplementedError("Pepper support comes later — use LocalRobot for now.")

    def say(self, text, bpm=None): ...
    def start_beat(self, bpm): ...
    def stop_beat(self): ...
    def set_bpm(self, bpm): ...
    def wait_for_downbeat(self): ...
    def cue_turn(self): ...
    def record(self, seconds): ...
    def do_action(self, action): ...