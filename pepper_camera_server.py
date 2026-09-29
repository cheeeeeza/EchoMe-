# -*- coding: utf-8 -*-
"""
Runs ON PEPPER. Streams Pepper's head camera to the laptop, which does the pose detection.

Copy it over and run it (from your laptop's PowerShell):
    scp pepper_camera_server.py nao@<PEPPER_IP>:/home/nao/
    ssh nao@<PEPPER_IP>
    python pepper_camera_server.py                 # then leave it running

Options:
    --port 5566      port the laptop connects to
    --res qvga|vga   qvga = 320x240 (smoother over Wi-Fi), vga = 640x480 (sharper)
    --fps 15
    --cam 0|1        0 = forehead camera (default), 1 = mouth camera (points down)
    --pitch 0.1      tilt the head to this angle (radians, + = look down) and hold it still,
                     so the camera doesn't wander while the child copies a move

Written for NAOqi 2.5 (Python 2.7 on the robot); also runs on Python 3 with the `qi` package.
"""
from __future__ import print_function

import argparse
import signal
import socket
import struct
import sys
import time

import qi

RESOLUTIONS = {"qvga": 1, "vga": 2}   # ALVideoDevice resolution ids
BGR_COLORSPACE = 13
MAGIC_JPEG, MAGIC_RAW = b"JPEG", b"RAW3"

try:                                  # JPEG makes frames ~10x smaller over Wi-Fi
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default="qvga")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--naoqi", default="tcp://127.0.0.1:9559")
    args = ap.parse_args()

    session = qi.Session()
    session.connect(args.naoqi)
    video = session.service("ALVideoDevice")

    if args.pitch is not None:
        hold_head_still(session, args.pitch)

    handle = video.subscribeCamera("rhythm_robot_cam", args.cam, RESOLUTIONS[args.res],
                                   BGR_COLORSPACE, args.fps)
    print("Camera ready (%s, %d fps, %s)." % (args.res, args.fps,
          "JPEG" if HAVE_CV2 else "raw frames - no OpenCV on robot, uses more Wi-Fi"))

    def cleanup(*_):
        try:
            video.unsubscribe(handle)
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print("Waiting for the laptop on port %d ... (Ctrl+C to stop)" % args.port)

    frame_gap = 1.0 / max(args.fps, 1)
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Laptop connected:", addr[0])
        try:
            while True:
                t0 = time.time()
                img = video.getImageRemote(handle)
                if img is None:
                    time.sleep(0.05)
                    continue
                w, h, data = img[0], img[1], img[6]
                if HAVE_CV2:
                    frame = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, w, 3)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    magic, payload = MAGIC_JPEG, jpg.tobytes()
                else:
                    magic, payload = MAGIC_RAW, bytes(data)
                conn.sendall(struct.pack("!4sIII", magic, w, h, len(payload)) + payload)
                time.sleep(max(0.0, frame_gap - (time.time() - t0)))
        except (socket.error, IOError):
            print("Laptop disconnected - waiting again.")
        finally:
            conn.close()


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


if __name__ == "__main__":
    main()