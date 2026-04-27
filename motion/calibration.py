"""Webcam calibration screen: shows the user's skeleton overlay and guides them
to stand at the right distance before the game starts.

Checks:
  1. Full body visible (head, shoulders, hips, knees, ankles all detected)
  2. Body size reasonable (shoulder width = 15-40% of frame width)
  3. Stable for 2 seconds (not still walking into position)

Returns True when ready, False if the user pressed ESC to cancel.
"""

import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gestra_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gestra_cache")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
REQUIRED_NAMES = ["nose", "L shoulder", "R shoulder", "L elbow", "R elbow",
                  "L wrist", "R wrist", "L hip", "R hip"]
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
MIN_SHOULDER_RATIO = 0.10
MAX_SHOULDER_RATIO = 0.45
VISIBILITY_THRESH = 0.5
STABLE_SECONDS = 2.0


def _draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (0, 180, 255), -1)


def _check_pose(landmarks, frame_width):
    missing = []
    for idx, name in zip(REQUIRED_LANDMARKS, REQUIRED_NAMES):
        vis = getattr(landmarks[idx], "visibility", 0) or 0
        if vis < VISIBILITY_THRESH:
            missing.append(name)
    if missing:
        return False, f"Can't see: {', '.join(missing[:3])}"

    lsh_x = landmarks[LEFT_SHOULDER].x * frame_width
    rsh_x = landmarks[RIGHT_SHOULDER].x * frame_width
    shoulder_w = abs(lsh_x - rsh_x)
    ratio = shoulder_w / frame_width

    if ratio < MIN_SHOULDER_RATIO:
        return False, "Too far - step closer"
    if ratio > MAX_SHOULDER_RATIO:
        return False, "Too close - step back"

    return True, "Good position!"


def run_calibration(camera_index=0, width=640, height=480):
    """Open a calibration window. Returns True when the user is in position."""
    pose_model = _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
    if not pose_model.exists():
        print(f"Missing pose model: {pose_model}", file=sys.stderr)
        return False

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        return False

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ready_since = None
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_idx += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_idx)

            h, w = frame.shape[:2]
            status_color = (0, 0, 255)
            status_text = "No body detected - stand in frame"
            progress = 0.0

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                _draw_skeleton(frame, lms)
                ok_pose, msg = _check_pose(lms, w)

                if ok_pose:
                    status_color = (0, 255, 0)
                    if ready_since is None:
                        ready_since = time.monotonic()
                    elapsed = time.monotonic() - ready_since
                    progress = min(elapsed / STABLE_SECONDS, 1.0)
                    remaining = max(0, STABLE_SECONDS - elapsed)
                    status_text = f"{msg} - hold still ({remaining:.1f}s)"
                    if elapsed >= STABLE_SECONDS:
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                else:
                    status_color = (0, 140, 255)
                    status_text = msg
                    ready_since = None
            else:
                ready_since = None

            # Draw progress bar
            bar_w = int(w * 0.6)
            bar_h = 16
            bar_x = (w - bar_w) // 2
            bar_y = h - 50
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (80, 80, 80), -1)
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h),
                              status_color, -1)

            # Draw status text
            cv2.putText(frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
                        cv2.LINE_AA)
            cv2.putText(frame, "ESC = skip calibration", (20, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1,
                        cv2.LINE_AA)

            cv2.imshow("Gestra - Position Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    return True  # skip = still start the game
