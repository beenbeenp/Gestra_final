"""Real-time webcam → MediaPipe Pose → LSTM → named action.

Runs in a background thread so the Pygame main loop stays responsive. The game
reads the latest predicted action via `latest_action()` (thread-safe).

Architecture:
    webcam frame → MediaPipe 33 landmarks → rolling 30-frame buffer
        → normalize (hip-center, shoulder-width)
        → ActionLSTM → {idle, punch, kick}
        → hip-velocity rule → override idle with forward/backward
        → 5-frame majority smoothing → final named action string

Usage:
    # From game main.py (see GESTRA_WEBCAM=1 path):
    from motion.pose_predictor import PosePredictor
    predictor = PosePredictor()
    predictor.start()
    action = predictor.latest_action()  # "idle" / "punch" / "kick" / "forward" / "backward"
    predictor.stop()

    # Standalone debug mode:
    python -m motion.pose_predictor --debug
"""

import argparse
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
import torch
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

# Lazy import to avoid circular deps when the game imports this module.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

ACTION_NAMES = ["idle", "punch", "kick"]
WINDOW_SIZE = 30
SMOOTHING_WINDOW = 3
HIP_VX_THRESHOLD = 0.004
ATTACK_COOLDOWN_FRAMES = 12  # ~0.4s at 30fps


def _normalize_frame(landmarks_33x3: np.ndarray) -> np.ndarray:
    out = landmarks_33x3.copy()
    hip_center = (out[LEFT_HIP] + out[RIGHT_HIP]) * 0.5
    out -= hip_center
    shoulder_width = max(
        np.linalg.norm(out[LEFT_SHOULDER, :2] - out[RIGHT_SHOULDER, :2]), 1e-6
    )
    out /= shoulder_width
    return out.astype(np.float32)


def _add_velocity_features_frame(frame_buffer):
    """Build (T, 297) from rolling frame buffer of (T, 99) position vectors."""
    seq = np.stack(list(frame_buffer), axis=0)  # (T, 99)
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


class PosePredictor:
    def __init__(
        self,
        model_path=None,
        pose_model_path=None,
        camera_index=0,
        width=640,
        height=480,
    ):
        if model_path is None:
            model_path = _PROJECT_ROOT / "motion" / "models" / "action_lstm.pt"
        if pose_model_path is None:
            pose_model_path = (
                _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
            )
        self._model_path = Path(model_path)
        self._pose_model_path = Path(pose_model_path)
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
        sys.path.insert(0, str(_PROJECT_ROOT))
        from ml.model import ActionLSTM
        from ml.dataset import FEAT_DIM

        model = ActionLSTM(input_dim=FEAT_DIM, num_classes=len(ACTION_NAMES))
        state = torch.load(self._model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
            print("PosePredictor: could not open camera", file=sys.stderr)
            self._running = False
            return

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._pose_model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frame_buffer = collections.deque(maxlen=WINDOW_SIZE)
        hip_x_buffer = collections.deque(maxlen=WINDOW_SIZE)
        frame_idx = 0
        attack_cooldown = 0
        last_raw_attack = None  # tracks the raw model output before edge logic

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
                    continue

                lms = result.pose_landmarks[0]
                raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                hip_x = (raw[LEFT_HIP, 0] + raw[RIGHT_HIP, 0]) * 0.5
                hip_x_buffer.append(hip_x)

                normed = _normalize_frame(raw)
                frame_buffer.append(normed.flatten())

                if len(frame_buffer) < WINDOW_SIZE:
                    continue

                # Build features with velocity + acceleration
                feat = _add_velocity_features_frame(frame_buffer)  # (30, 297)
                tensor = torch.from_numpy(feat).unsqueeze(0)  # (1, 30, 297)
                with torch.no_grad():
                    logits = model(tensor)
                pred_idx = int(logits.argmax(1).item())
                action = ACTION_NAMES[pred_idx]

                # Hip-velocity rule: override idle with forward/backward
                if action == "idle" and len(hip_x_buffer) >= 10:
                    recent = list(hip_x_buffer)[-10:]
                    vx = (recent[-1] - recent[0]) / len(recent)
                    if vx > HIP_VX_THRESHOLD:
                        action = "forward"
                    elif vx < -HIP_VX_THRESHOLD:
                        action = "backward"

                # Edge trigger: fire attack once on rising edge, then cooldown.
                # No smoothing for attacks — they must be snappy.
                if action in ("punch", "kick"):
                    if attack_cooldown > 0:
                        action = "idle"
                    elif last_raw_attack == action:
                        action = "idle"
                    else:
                        attack_cooldown = ATTACK_COOLDOWN_FRAMES
                    last_raw_attack = ACTION_NAMES[pred_idx]
                else:
                    last_raw_attack = None

                if attack_cooldown > 0:
                    attack_cooldown -= 1

                with self._lock:
                    self._action = action

        cap.release()


def _debug_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    predictor = PosePredictor(camera_index=args.camera)
    predictor.start()
    print("PosePredictor running. Press Ctrl+C to stop.")
    try:
        while True:
            action = predictor.latest_action()
            print(f"\r  action: {action:12s}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        predictor.stop()


if __name__ == "__main__":
    _debug_main()
