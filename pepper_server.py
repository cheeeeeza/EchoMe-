# -*- coding: utf-8 -*-
"""
Runs ON PEPPER. Two jobs:
  1. Streams Pepper's head camera to the laptop (port 5566) for pose detection.
  2. Takes commands from the laptop (port 5567): speak, run the beat, play the
     "your turn" chime, do gestures (wave), and set the output volume. The beat runs here
     on the robot, so Wi-Fi delays can't make it wobble.

Copy it over and run it (from your laptop's PowerShell):
    scp pepper_server.py nao@<PEPPER_IP>:/home/nao/
    ssh nao@<PEPPER_IP>
    python pepper_server.py --pitch 0.1            # then leave it running

Options:
    --res qvga|vga   camera: qvga = 320x240 (smoother over Wi-Fi), vga = 640x480
    --fps 15         camera frame rate
    --cam 0|1        0 = forehead camera, 1 = mouth camera (points down)
    --pitch 0.1      hold the head still at this tilt (radians, + = look down)
    --gain 30        beat loudness, 0-100
    --volume 0.8     speech volume, 0-1

Written for NAOqi 2.5 (Python 2.7 on the robot); also runs on Python 3 with `qi`.
"""
from __future__ import print_function

import argparse
import json
import signal
import socket
import struct
import sys
import threading
import time
import traceback

import qi

RESOLUTIONS = {"qvga": 1, "vga": 2}
BGR_COLORSPACE = 13
BEATS_PER_BAR = 4
PY2 = sys.version_info[0] == 2

try:                                  # JPEG makes frames ~10x smaller over Wi-Fi
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def _s(text):
    """NAOqi on Python 2 wants UTF-8 byte strings."""
    if PY2 and not isinstance(text, str):
        return text.encode("utf-8")
    return text


def call_async(fn, *args):
    """Fire a NAOqi call without waiting for it."""
    try:
        return fn(*args, _async=True)
    except TypeError:                 # very old qi: fall back to a thread
        t = threading.Thread(target=fn, args=args)
        t.daemon = True
        t.start()


def output_level(volume):
    """The laptop sends 0.0-1.0; ALAudioDevice wants an int 0-100 ("Volume [0-100]")."""
    return int(max(0, min(100, round(float(volume) * 100))))


# ------------------------------------------------------------------ beat
class Metronome(object):
    def __init__(self, audio, gain):
        self.audio, self.gain = audio, gain
        self.bpm = 80
        self.running = False
        self.beat_index = 0            # index of the NEXT beat to play
        self.next_time = 0.0
        self.lock = threading.Lock()

    def start(self, bpm):
        with self.lock:
            self.bpm = bpm
            if self.running:
                return
            self.running = True
            self.beat_index = 0
            self.next_time = time.time() + 0.1
        t = threading.Thread(target=self._run)
        t.daemon = True
        t.start()

    def stop(self):
        with self.lock:
            self.running = False

    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm

    def seconds_to_next_bar(self):
        with self.lock:
            if not self.running:
                return 0.0
            beats_left = (-self.beat_index) % BEATS_PER_BAR
            return max(0.0, self.next_time - time.time()) + beats_left * 60.0 / self.bpm

    def _run(self):
        while True:
            with self.lock:
                if not self.running:
                    return
                t, idx = self.next_time, self.beat_index
            delay = t - time.time()
            if delay > 0:
                time.sleep(delay)
            with self.lock:
                if not self.running:
                    return
                self.beat_index += 1
                self.next_time = t + 60.0 / self.bpm
            accent = idx % BEATS_PER_BAR == 0
            try:
                call_async(self.audio.playSine, 1500 if accent else 1000, self.gain, 0, 0.04)
            except Exception:
                pass


# ------------------------------------------------------------------ gestures
ARM = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
BUILTIN_WAVE = "animations/Stand/Gestures/Hey_1"     # Pepper's standard "hey" wave


class Gestures(object):
    """Moves Haku's arms. Runs in the background so Haku can talk at the same time."""

    def __init__(self, session):
        self.motion = session.service("ALMotion")
        try:
            self.anim = session.service("ALAnimationPlayer")
        except Exception:
            self.anim = None
        self.busy = threading.Lock()

    def run(self, name):
        if name != "wave":
            raise ValueError("no Pepper gesture for '%s' yet" % name)
        print("Gesture: %s" % name)
        try:
            if not self.motion.robotIsWakeUp():
                print("  ! Haku is resting (motors off), so it can't move. "
                      "Run haku_check.py --wake, then restart this script.")
                return
        except Exception:
            pass
        if not self.busy.acquire(False):         # already moving: skip rather than queue up
            print("  (still doing the last gesture - skipped)")
            return
        t = threading.Thread(target=self._wave)
        t.daemon = True
        t.start()

    def _wave(self):
        try:
            if self.anim is not None:
                try:
                    self.anim.run(BUILTIN_WAVE)
                    print("  done (built-in wave)")
                    return
                except Exception as e:
                    print("(built-in wave unavailable: %s - using custom wave)" % e)
                    self.anim = None                 # don't try again every time
            self._custom_wave()
            print("  done (custom wave)")
        except Exception:
            traceback.print_exc()
        finally:
            self.busy.release()

    def _custom_wave(self):
        """Right arm up, forearm waves side to side 3 times, then back to where it was."""
        self.motion.setStiffnesses("RArm", 1.0)
        start = self.motion.getAngles(ARM, True)
        up = [-1.0, -0.35, 1.3, 0.5, -0.3, 0.9]     # arm raised, hand open
        wave_in = list(up)
        wave_in[3] = 1.1                             # bend elbow = forearm swings in
        frames = [up, wave_in, up, wave_in, up, wave_in, up, start]
        times = [0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 3.6]
        angle_lists = [[f[j] for f in frames] for j in range(len(ARM))]
        time_lists = [times] * len(ARM)
        self.motion.angleInterpolation(ARM, angle_lists, time_lists, True)


# ------------------------------------------------------------------ commands
def serve_commands(port, tts, audio, metro, gain, gestures, audiodev):
    server = _listen(port)
    while True:
        conn, addr = server.accept()
        print("Laptop connected for speech/beat:", addr[0])
        reader = conn.makefile("r")
        try:
            for line in reader:
                reply = _run_command(line, tts, audio, metro, gain, gestures, audiodev)
                conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        except (socket.error, IOError):
            pass
        finally:
            metro.stop()
            conn.close()
            print("Speech/beat connection closed - waiting again.")


def _run_command(line, tts, audio, metro, gain, gestures, audiodev):
    try:
        cmd = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "bad json"}
    op = cmd.get("op")
    try:
        if op == "say":
            if cmd.get("downbeat"):
                time.sleep(metro.seconds_to_next_bar())
            speed = int(cmd.get("speed", 100))
            tts.say(_s(u"\\rspd=%d\\%s" % (speed, cmd["text"])))
        elif op == "beat_start":
            metro.start(int(cmd["bpm"]))
        elif op == "beat_stop":
            metro.stop()
        elif op == "set_bpm":
            metro.set_bpm(int(cmd["bpm"]))
        elif op == "chime":
            audio.playSine(880, min(100, gain + 20), 0, 0.12)
            audio.playSine(1320, min(100, gain + 20), 0, 0.18)
        elif op == "gesture":
            if gestures is None:
                return {"ok": False, "error": "gestures unavailable on this robot"}
            gestures.run(cmd["name"])
        elif op == "set_volume":
            level = output_level(cmd["volume"])
            audiodev.setOutputVolume(level)
            print("Laptop set the output volume to %d/100" % level)
        elif op == "ping":
            pass
        else:
            return {"ok": False, "error": "unknown op %s" % op}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}


# ------------------------------------------------------------------ camera
JPEG_QUALITY_FLAG = getattr(cv2, "IMWRITE_JPEG_QUALITY", 1) if HAVE_CV2 else 1


def _as_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, list):                  # some NAOqi versions return a list of ints
        return bytes(bytearray(data))
    return bytes(data)


def _array_bytes(arr):
    # numpy < 1.9 (older robots) only has tostring()
    return arr.tobytes() if hasattr(arr, "tobytes") else arr.tostring()


def encode_frame(w, h, data, state):
    raw = _as_bytes(data)
    if len(raw) != w * h * 3:
        raise ValueError("frame is %d bytes, expected %d (%dx%dx3)" % (len(raw), w * h * 3, w, h))
    if HAVE_CV2 and not state.get("raw"):
        try:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            ok, jpg = cv2.imencode(".jpg", frame, [JPEG_QUALITY_FLAG, 80])
            if ok:
                return b"JPEG", _array_bytes(jpg)
        except Exception:
            traceback.print_exc()
            print("JPEG compression isn't working on this robot - sending raw frames instead.")
            state["raw"] = True
    return b"RAW3", raw


def serve_camera(port, video, handle, fps):
    server = _listen(port)
    frame_gap = 1.0 / max(fps, 1)
    state = {}
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Laptop connected for camera:", addr[0])
        errors = 0
        try:
            while True:
                t0 = time.time()
                try:
                    img = video.getImageRemote(handle)
                    if img is None:
                        time.sleep(0.05)
                        continue
                    magic, payload = encode_frame(img[0], img[1], img[6], state)
                    errors = 0
                except Exception:                  # a bad frame must never kill the camera
                    errors += 1
                    if errors in (1, 20):
                        traceback.print_exc()
                    time.sleep(0.1)
                    continue
                conn.sendall(struct.pack("!4sIII", magic, img[0], img[1], len(payload)) + payload)
                time.sleep(max(0.0, frame_gap - (time.time() - t0)))
        except (socket.error, IOError):
            print("Camera connection closed - waiting again.")
        except Exception:
            traceback.print_exc()
        finally:
            conn.close()


# ------------------------------------------------------------------ setup
def _listen(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    return s


def hold_head_still(session, pitch):
    """Stop Pepper glancing around (it would move the camera) and point the head."""
    try:
        session.service("ALBasicAwareness").pauseAwareness()
    except Exception as e:
        print("(couldn't pause basic awareness: %s)" % e)
    try:
        motion = session.service("ALMotion")
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, pitch], 0.1)
    except Exception as e:
        print("(couldn't set head angle: %s)" % e)


def _thread(target, *args):
    t = threading.Thread(target=target, args=args)
    t.daemon = True
    t.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5566, help="camera port")
    ap.add_argument("--cmd-port", type=int, default=5567, help="speech/beat port")
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default="qvga")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--gain", type=int, default=30)
    ap.add_argument("--volume", type=float, default=0.8)
    ap.add_argument("--naoqi", default="tcp://127.0.0.1:9559")
    args = ap.parse_args()

    session = qi.Session()
    session.connect(args.naoqi)
    video = session.service("ALVideoDevice")
    audio = session.service("ALAudioPlayer")
    tts = session.service("ALTextToSpeech")
    audiodev = session.service("ALAudioDevice")
    try:
        tts.setLanguage("English")
        tts.setVolume(args.volume)
    except Exception as e:
        print("(couldn't set speech language/volume: %s)" % e)
    if args.pitch is not None:
        hold_head_still(session, args.pitch)

    handle = video.subscribeCamera("rhythm_robot_cam", args.cam, RESOLUTIONS[args.res],
                                   BGR_COLORSPACE, args.fps)
    metro = Metronome(audio, args.gain)
    try:
        gestures = Gestures(session)
    except Exception as e:
        print("(gestures unavailable: %s)" % e)
        gestures = None

    def cleanup(*_):
        metro.stop()
        try:
            video.unsubscribe(handle)
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    _thread(serve_camera, args.port, video, handle, args.fps)
    _thread(serve_commands, args.cmd_port, tts, audio, metro, args.gain, gestures, audiodev)
    print("Camera ready (%s, %d fps, %s)." % (args.res, args.fps, "JPEG" if HAVE_CV2 else "raw frames"))
    print("Waiting for the laptop on ports %d (camera) and %d (speech/beat)... Ctrl+C to stop"
          % (args.port, args.cmd_port))
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()