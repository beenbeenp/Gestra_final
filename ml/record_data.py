"""Record your own pose data for training.

Guides you through each action with on-screen prompts, records MediaPipe
landmarks, and saves labeled .npz files ready for training.

Usage:
    cd Gestra_final
    .venv/bin/python -m ml.record_data
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

ACTIONS = [
    ("idle", "Stand still, arms relaxed", 8),
    ("lpunch", "Raise LEFT arm above shoulder repeatedly", 12),
    ("idle", "Stand still again, arms relaxed", 6),
    ("rpunch", "Raise RIGHT arm above shoulder repeatedly", 12),
    ("idle", "Stand still, rest", 6),
    ("forward", "Lean RIGHT repeatedly", 10),
    ("backward", "Lean LEFT repeatedly", 10),
    ("idle", "Final idle - stand still", 6),
]


def _draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 180, 255), -1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=Path, default=_PROJECT_ROOT / "data" / "personal")
    p.add_argument("--person", type=str, default=None,
                   help="Person name/ID. Data saved to <out>/<person>/. "
                        "Multiple people can record separately, then train together.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    if args.person:
        args.out = args.out / args.person

    pose_model = _PROJECT_ROOT / "motion" / "models" / "pose_landmarker_lite.task"
    if not pose_model.exists():
        sys.exit(f"Missing: {pose_model}")

    args.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Cannot open camera")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("=== Gestra Data Recorder ===")
    print(f"Output: {args.out}")
    print("Press SPACE in the preview window to start each action.")
    print("Press ESC at any time to abort.\n")

    frame_idx = 0
    all_segments = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for action_name, instruction, duration in ACTIONS:
            # Countdown screen
            print(f"\nNext: [{action_name}] {instruction} ({duration}s)")
            waiting = True
            while waiting:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_idx += 1
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_idx)
                if result.pose_landmarks:
                    _draw_skeleton(frame, result.pose_landmarks[0])

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
                cv2.putText(frame, f"NEXT: {instruction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to start, ESC to quit", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.imshow("Gestra Recorder", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Aborted.")
                    return
                if key == 32:
                    waiting = False

            # Record
            landmarks_buf = []
            valid_buf = []
            start = time.monotonic()
            print(f"  Recording [{action_name}]...", end="", flush=True)

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
                    xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    landmarks_buf.append(xyz)
                    valid_buf.append(True)
                else:
                    landmarks_buf.append(np.zeros((33, 3), dtype=np.float32))
                    valid_buf.append(False)

                elapsed = time.monotonic() - start
                remaining = max(0, duration - elapsed)
                h, w = frame.shape[:2]
                # Progress bar
                bar_w = int(w * 0.8)
                bar_x = (w - bar_w) // 2
                progress = elapsed / duration
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(frame, f"RECORDING: {action_name} ({remaining:.1f}s)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (bar_x, h - 30), (bar_x + bar_w, h - 14), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, h - 30), (bar_x + int(bar_w * progress), h - 14), (0, 0, 255), -1)
                cv2.imshow("Gestra Recorder", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    print(" Aborted.")
                    return

            n_valid = sum(valid_buf)
            print(f" {len(landmarks_buf)} frames, {n_valid} with pose")
            all_segments.append({
                "action": action_name,
                "landmarks": np.stack(landmarks_buf),
                "valid": np.array(valid_buf, dtype=bool),
            })

    cap.release()
    cv2.destroyAllWindows()

    # Save per-action .npz files
    counts = {}
    for seg in all_segments:
        name = seg["action"]
        counts[name] = counts.get(name, 0) + 1
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"personal_{name}_{counts[name]:02d}.npz"
        np.savez_compressed(out_path,
                            landmarks=seg["landmarks"],
                            valid=seg["valid"],
                            fps=30.0)
        print(f"  Saved {out_path} ({seg['landmarks'].shape[0]} frames)")

    print(f"\nDone! Recorded {len(all_segments)} segments to {args.out}")
    print("Now run:  .venv/bin/python -m ml.train_personal")


if __name__ == "__main__":
    main()
