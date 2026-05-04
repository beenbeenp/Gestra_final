"""Rule-based upper-body action detector. No ML, no training data.

Works at laptop distance (~0.5-1.0m), only needs shoulders/elbows/wrists visible.
Detects actions by simple geometric rules on MediaPipe landmarks:

  punch:    wrist rises above shoulder height + wrist velocity > threshold
  forward:  shoulder center shifts right
  backward: shoulder center shifts left
  idle:     none of the above

All detection is instantaneous (single-frame + short velocity buffer).
"""

import collections
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gestra_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gestra_cache")

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# Thresholds (tuned for seated user facing laptop webcam)
PUNCH_RAISE_THRESH = 0.0       # wrist at or above shoulder center (normalized by shoulder width)
PUNCH_SPEED_THRESH = 0.04
TILT_THRESH = 0.15             # shoulder center horizontal shift — wider dead zone for idle stability
COOLDOWN_FRAMES = 12


class UpperBodyDetector:
    """Thread-safe real-time upper-body action detector."""

    def __init__(self, camera_index=0, width=640, height=480):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._action = "idle"
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def latest_action(self):
        with self._lock:
            return self._action

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self):
        pose_model = _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
        if not pose_model.exists():
            print(f"UpperBodyDetector: missing {pose_model}", file=sys.stderr)
            self._running = False
            return

        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
            print("UpperBodyDetector: could not open camera", file=sys.stderr)
            self._running = False
            return

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frame_idx = 0
        prev_wrists = None
        sh_x_buf = collections.deque(maxlen=10)
        baseline_sh_x = None
        cooldown = 0
        action_buf = collections.deque(maxlen=7)  # smoothing buffer

        with PoseLandmarker.create_from_options(options) as landmarker:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame_idx += 1
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_idx)

                if not result.pose_landmarks:
                    prev_wrists = None
                    continue

                lms = result.pose_landmarks[0]

                # Extract key points (normalized 0-1 image space)
                ls = np.array([lms[L_SHOULDER].x, lms[L_SHOULDER].y])
                rs = np.array([lms[R_SHOULDER].x, lms[R_SHOULDER].y])
                lw = np.array([lms[L_WRIST].x, lms[L_WRIST].y])
                rw = np.array([lms[R_WRIST].x, lms[R_WRIST].y])

                # Shoulder width as scale reference
                sh_width = max(np.linalg.norm(ls - rs), 0.01)
                sh_center_x = (ls[0] + rs[0]) * 0.5
                sh_center_y = (ls[1] + rs[1]) * 0.5

                # Track shoulder center x for tilt detection
                # Baseline updates slowly so returning to center = idle, not opposite direction
                sh_x_buf.append(sh_center_x)
                if baseline_sh_x is None and len(sh_x_buf) >= 8:
                    baseline_sh_x = np.mean(list(sh_x_buf))
                elif baseline_sh_x is not None:
                    baseline_sh_x = baseline_sh_x * 0.95 + sh_center_x * 0.05

                # Wrist distances from shoulders (normalized by shoulder width)
                l_dist = np.linalg.norm(lw - ls) / sh_width
                r_dist = np.linalg.norm(rw - rs) / sh_width

                # Wrist height relative to shoulders (negative = above)
                l_height = (lw[1] - sh_center_y) / sh_width
                r_height = (rw[1] - sh_center_y) / sh_width

                # Wrist speed
                wrist_speed = 0.0
                if prev_wrists is not None:
                    dl = np.linalg.norm(lw - prev_wrists[:2]) / sh_width
                    dr = np.linalg.norm(rw - prev_wrists[2:]) / sh_width
                    wrist_speed = max(dl, dr)
                prev_wrists = np.concatenate([lw, rw])

                # Detect action
                action = "idle"

                if cooldown > 0:
                    cooldown -= 1
                else:
                    # Punch: either wrist raised above shoulder + moving fast
                    if (l_height < PUNCH_RAISE_THRESH or r_height < PUNCH_RAISE_THRESH) and wrist_speed > PUNCH_SPEED_THRESH:
                        if l_height < r_height:
                            action = "lpunch"
                        else:
                            action = "rpunch"
                        cooldown = COOLDOWN_FRAMES

                # Forward/backward: body tilt left/right
                if action == "idle" and baseline_sh_x is not None:
                    tilt = (sh_center_x - baseline_sh_x) / sh_width
                    if tilt > TILT_THRESH:
                        action = "forward"
                    elif tilt < -TILT_THRESH:
                        action = "backward"

                # Smoothing: require majority vote over last 5 frames
                # Punches bypass smoothing (they need to be responsive)
                if action in ("lpunch", "rpunch"):
                    smoothed = action
                else:
                    action_buf.append(action)
                    counts = collections.Counter(action_buf)
                    smoothed = counts.most_common(1)[0][0]

                with self._lock:
                    self._action = smoothed

        cap.release()


def _debug_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    det = UpperBodyDetector(camera_index=args.camera)
    det.start()
    print("UpperBodyDetector running. Ctrl+C to stop.")
    try:
        while True:
            print(f"\r  action: {det.latest_action():12s}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        det.stop()


if __name__ == "__main__":
    _debug_main()
