"""Quick pre-game recording: ~30s guided session to capture the current user's
motion patterns. Runs in an OpenCV window between calibration and game start.

Returns the output directory path if recording completed, or None if skipped.
"""

import os
import sys
import time
from pathlib import Path

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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUICK_ACTIONS = [
    ("idle",     "Stay still",       5),
    ("lpunch",   "Raise LEFT arm",   4),
    ("idle",     "Stay still",       3),
    ("rpunch",   "Raise RIGHT arm",  4),
    ("idle",     "Stay still",       3),
    ("forward",  "Lean RIGHT",       4),
    ("backward", "Lean LEFT",        4),
    ("idle",     "Stay still",       3),
]


def _draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 180, 255), -1)


def run_quick_record(camera_index=0, width=640, height=480):
    """Show a prompt, then run a ~30s guided recording session.

    Returns the output directory (Path) if completed, or None if skipped.
    """
    pose_model = _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
    if not pose_model.exists():
        print(f"quick_record: missing {pose_model}", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("quick_record: could not open camera", file=sys.stderr)
        return None

    # Prompt screen: SPACE to record, ESC to skip
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "QUICK CALIBRATION", (w // 2 - 180, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Record your moves to improve accuracy (~30s)",
                    (w // 2 - 250, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE = Start    ESC = Skip",
                    (w // 2 - 160, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gestra - Quick Record", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
        if key == 32:  # SPACE
            break

    # Set up output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "data" / "personal" / f"quick_{timestamp}"

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    all_segments = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for action_name, instruction, duration in QUICK_ACTIONS:
            landmarks_buf = []
            valid_buf = []
            start = time.monotonic()

            while time.monotonic() - start < duration:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_idx += 1
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_idx)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    _draw_skeleton(frame, lms)
                    xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                                   dtype=np.float32)
                    landmarks_buf.append(xyz)
                    valid_buf.append(True)
                else:
                    landmarks_buf.append(np.zeros((33, 3), dtype=np.float32))
                    valid_buf.append(False)

                elapsed = time.monotonic() - start
                remaining = max(0, duration - elapsed)
                h, w = frame.shape[:2]

                # Action label + progress bar
                color = (0, 0, 255) if action_name != "idle" else (0, 200, 0)
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(frame, f"{instruction} ({remaining:.1f}s)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                bar_w = int(w * 0.8)
                bar_x = (w - bar_w) // 2
                progress = elapsed / duration
                cv2.rectangle(frame, (bar_x, h - 25),
                              (bar_x + bar_w, h - 12), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, h - 25),
                              (bar_x + int(bar_w * progress), h - 12), color, -1)

                cv2.imshow("Gestra - Quick Record", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

            n_valid = sum(valid_buf)
            all_segments.append({
                "action": action_name,
                "landmarks": np.stack(landmarks_buf) if landmarks_buf else np.zeros((0, 33, 3)),
                "valid": np.array(valid_buf, dtype=bool),
            })
            print(f"  [{action_name}] {len(landmarks_buf)} frames, {n_valid} valid")

    cap.release()
    cv2.destroyAllWindows()

    # Save per-action npz files
    counts = {}
    for seg in all_segments:
        name = seg["action"]
        if len(seg["landmarks"]) == 0:
            continue
        counts[name] = counts.get(name, 0) + 1
        action_dir = out_dir / name
        action_dir.mkdir(parents=True, exist_ok=True)
        out_path = action_dir / f"quick_{name}_{counts[name]:02d}.npz"
        np.savez_compressed(out_path, landmarks=seg["landmarks"],
                            valid=seg["valid"], fps=30.0)

    print(f"Quick recording saved to {out_dir}")
    return out_dir
