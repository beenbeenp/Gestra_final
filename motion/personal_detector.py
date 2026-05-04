"""Personal-model detector: uses a TCN trained on your own recorded data.

Same interface as UpperBodyDetector (latest_action() returns a string),
but uses the personal TCN model instead of geometric rules.
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
import torch
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from ml.model import ActionTCN

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
NUM_JOINTS = len(UPPER_BODY_JOINTS)
WINDOW = 20
COOLDOWN_FRAMES = 12


def _normalize_frame(lm):
    out = lm.copy()
    shoulder_center = (out[LEFT_SHOULDER] + out[RIGHT_SHOULDER]) * 0.5
    out -= shoulder_center
    sw = max(np.linalg.norm(out[LEFT_SHOULDER, :2] - out[RIGHT_SHOULDER, :2]), 1e-6)
    out /= sw
    return out[UPPER_BODY_JOINTS].astype(np.float32)


def _add_vel_acc(seq):
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


class PersonalModelDetector:
    def __init__(self, camera_index=0, width=640, height=480, model_path=None):
        if model_path is None:
            model_path = _PROJECT_ROOT / "motion" / "models" / "action_personal.pt"
        self._model_path = Path(model_path)
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

    def _loop(self):
        checkpoint = torch.load(self._model_path, map_location="cpu", weights_only=False)
        action_names = checkpoint["action_names"]
        num_classes = checkpoint["num_classes"]

        model = ActionTCN(input_dim=NUM_JOINTS * 3 * 3, num_classes=num_classes,
                          channels=(64, 64), kernel_size=5, dropout=0.0)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        pose_model = _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
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

        buf = collections.deque(maxlen=WINDOW)
        frame_idx = 0
        cooldown = 0

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
                normed = _normalize_frame(raw)
                buf.append(normed.flatten())

                if len(buf) < WINDOW:
                    continue

                seq = np.stack(list(buf), axis=0)
                feat = _add_vel_acc(seq)
                tensor = torch.from_numpy(feat).unsqueeze(0)
                with torch.no_grad():
                    pred = model(tensor).argmax(1).item()
                action = action_names[pred]

                # Cooldown for attacks
                if cooldown > 0:
                    cooldown -= 1
                    if action in ("lpunch", "rpunch"):
                        action = "idle"
                elif action in ("lpunch", "rpunch"):
                    cooldown = COOLDOWN_FRAMES

                with self._lock:
                    self._action = action

        cap.release()
