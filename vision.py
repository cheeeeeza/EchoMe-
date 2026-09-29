"""
Camera window: the child's mirrored camera (Pepper's head camera or the laptop webcam) on the left, the robot (a stick
figure standing in for Pepper) on the right. Pose detection uses MediaPipe.
Press Q in the window to stop the session.
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

import config
from actions import ACTIONS, Detector, demo_frame

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
PANEL_W, PANEL_H = 260, 300
_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


@dataclass
class ActionResult:
    action: str
    done: bool
    time_s: float | None        # seconds from "your turn" until the move was recognised
    person_seen: bool           # False = the child wasn't visible in the camera at all


class WebcamSource:
    def __init__(self, cv2, index: int = 0):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError("Couldn't open the laptop webcam.")

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.release()


class PepperCameraSource:
    """Receives frames streamed by pepper_camera_server.py (running on the robot)."""

    def __init__(self, cv2, ip: str, port: int):
        self.cv2, self.addr = cv2, (ip, port)
        self._latest, self._fresh = None, False
        self._cond = threading.Condition()
        self._stop = False
        self.sock = self._connect()
        threading.Thread(target=self._receive, daemon=True).start()
        ok, _ = self.read(timeout=5.0)
        if not ok:
            raise RuntimeError("Connected to Pepper but no frames arrived.")
        print(f"📷 Receiving Pepper's camera from {ip}:{port}")

    def _connect(self):
        try:
            return socket.create_connection(self.addr, timeout=5)
        except OSError as e:
            raise RuntimeError(
                f"Couldn't reach Pepper's camera at {self.addr[0]}:{self.addr[1]} ({e}). "
                "Is pepper_camera_server.py running on the robot, and is PEPPER_IP right?") from e

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Pepper closed the camera stream")
            buf += chunk
        return bytes(buf)

    def _receive(self):
        while not self._stop:
            try:
                magic, w, h, size = struct.unpack("!4sIII", self._recv_exact(16))
                payload = self._recv_exact(size)
                if magic == b"JPEG":
                    frame = self.cv2.imdecode(np.frombuffer(payload, np.uint8), self.cv2.IMREAD_COLOR)
                else:
                    frame = np.frombuffer(payload, np.uint8).reshape(h, w, 3)
                with self._cond:
                    self._latest, self._fresh = frame, True   # older frames are simply dropped
                    self._cond.notify_all()
            except Exception as e:
                if self._stop:
                    return
                print(f"\n⚠️  Pepper camera stream dropped ({e}) — reconnecting...")
                time.sleep(1.0)
                try:
                    self.sock = self._connect()
                except RuntimeError:
                    pass

    def read(self, timeout: float = 0.5):
        """Newest frame since the last read (waits briefly for one)."""
        with self._cond:
            if not self._fresh:
                self._cond.wait(timeout)
            if not self._fresh:
                return False, None
            self._fresh = False
            return True, self._latest.copy()

    def close(self):
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


class Vision:
    def __init__(self, source: str = "webcam", pepper_ip: str = "", pepper_port: int = 5566):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.cv2, self.mp = cv2, mp
        # connect to the camera first, so a missing robot fails fast and cleanly
        if source == "pepper":
            self.cap = PepperCameraSource(cv2, pepper_ip, pepper_port)
        else:
            self.cap = WebcamSource(cv2)
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose model (~6 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

        self._t0 = time.monotonic()
        self._last_ts = 0
        self.window = "Rhythm Robot  (Q = stop)"

    # ---------------------------------------------------------------- pose
    def _pose(self, frame_bgr):
        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        ts = max(self._last_ts + 1, int((time.monotonic() - self._t0) * 1000))
        self._last_ts = ts
        res = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.pose_landmarks:
            return None
        return [(p.x, p.y, p.visibility if p.visibility is not None else 1.0)
                for p in res.pose_landmarks[0]]

    # ---------------------------------------------------------------- drawing
    def _draw(self, frame, lm, robot_action, status, sub, colour, t):
        cv2 = self.cv2
        frame = cv2.flip(frame, 1)                      # mirror, like a mirror
        h, w = frame.shape[:2]
        if lm:
            pts = {i: (int((1 - lm[i][0]) * w), int(lm[i][1] * h)) for i in range(len(lm))}
            for a, b in _BONES:
                if lm[a][2] > 0.4 and lm[b][2] > 0.4:
                    cv2.line(frame, pts[a], pts[b], (80, 220, 120), 3)
        else:
            cv2.putText(frame, "Step back so I can see your arms", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        panel = np.full((h, PANEL_W, 3), 245, np.uint8)
        oy = max(0, (h - PANEL_H) // 2)
        self._draw_robot(panel, robot_action, t, oy)
        cv2.putText(panel, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        if sub:
            cv2.putText(panel, sub, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)
        cv2.imshow(self.window, np.hstack([frame, panel]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            raise KeyboardInterrupt

    def _draw_robot(self, img, action, t, oy):
        cv2 = self.cv2
        o = lambda p: (p[0], p[1] + oy)
        body, arm, face = (120, 120, 140), (70, 110, 200), (255, 255, 255)
        cv2.circle(img, o((130, 62)), 30, body, -1)                      # head
        cv2.circle(img, o((119, 58)), 6, face, -1)
        cv2.circle(img, o((141, 58)), 6, face, -1)                      # eyes
        cv2.ellipse(img, o((130, 70)), (10, 6), 0, 0, 180, face, 2)     # smile
        cv2.line(img, o((130, 92)), o((130, 205)), body, 10)            # torso
        cv2.line(img, o((100, 110)), o((160, 110)), body, 10)           # shoulders
        cv2.line(img, o((130, 205)), o((110, 270)), body, 8)
        cv2.line(img, o((130, 205)), o((150, 270)), body, 8)            # legs
        le, lw, re, rw = demo_frame(action, t)
        for sh, el, wr in (((100, 110), le, lw), ((160, 110), re, rw)):
            cv2.line(img, o(sh), o(el), arm, 8)
            cv2.line(img, o(el), o(wr), arm, 8)
            cv2.circle(img, o(wr), 8, arm, -1)
        if action:
            cv2.putText(img, action.replace("_", " "), (12, oy + PANEL_H - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, arm, 2)

    # ---------------------------------------------------------------- public
    def robot_turn(self, action: str | None, speaking_thread, min_seconds: float = 1.0):
        """Animate the robot doing `action` while it speaks (speech runs in a thread)."""
        start = time.monotonic()
        while speaking_thread.is_alive() or time.monotonic() - start < min_seconds:
            ok, frame = self.cap.read()
            if not ok:
                self.cv2.waitKey(10)
                continue
            t = time.monotonic() - start
            self._draw(frame, self._pose(frame), action, "Watch me!", "", (200, 110, 70), t)

    def watch(self, action: str | None, seconds: float, prompt: str = "Your turn!") -> ActionResult | None:
        """Child's turn. Shows the camera + countdown; checks `action` if there is one."""
        det = Detector(action) if action else None
        seen = False
        start = time.monotonic()
        while (t := time.monotonic() - start) < seconds:
            ok, frame = self.cap.read()
            if not ok:
                self.cv2.waitKey(10)
                continue
            lm = self._pose(frame)
            seen |= lm is not None
            if det:
                det.update(lm, t)
            if det and det.done:
                status, colour = "Great move!", (60, 170, 60)
            else:
                status, colour = prompt, (70, 110, 200)
            show = action if (det and config.SHOW_ACTION_DURING_TURN and not det.done) else None
            self._draw(frame, lm, show, status, f"{seconds - t:.1f}s", colour, t)
        if not det:
            return None
        return ActionResult(action, det.done, round(det.done_at, 2) if det.done else None, seen)

    def idle(self, text: str = ""):
        ok, frame = self.cap.read()
        if ok:
            self._draw(frame, None, None, text, "", (120, 120, 120), 0)

    def close(self):
        self.cap.release()
        self.landmarker.close()
        self.cv2.destroyAllWindows()