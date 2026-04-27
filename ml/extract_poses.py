"""Extract MediaPipe Pose 33-landmark sequences from video clips.

For each video file under `data/videos/<class>/`, runs MediaPipe Pose Landmarker
on every frame and writes a .npz file under `data/poses/<class>/` containing:

    landmarks : (T, 33, 3) float32, normalized image-space (x, y, z)
    visibility: (T, 33)    float32
    valid     : (T,)       bool, True iff a pose was detected on that frame
    fps       : float, source video FPS

Clips where fewer than --min-detect-rate of frames yield a pose are skipped —
they would only inject noise into training. The same MediaPipe model
(motion/models/pose_landmarker_lite.task) is reused at inference time so the
training and runtime feature distributions match.

Re-running is idempotent: skips classes' clips that already have valid output.
"""

import argparse
import os
import sys
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
)
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path, default=Path("data/videos"))
    p.add_argument("--out", type=Path, default=Path("data/poses"))
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "motion"
        / "models"
        / "pose_landmarker_lite.task",
    )
    p.add_argument("--min-detect-rate", type=float, default=0.5,
                   help="Reject clips where the per-frame pose-detection rate is "
                        "below this fraction.")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Cap frames per clip; HMDB51 clips are usually <100.")
    return p.parse_args()


def extract_clip(landmarker, video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarks_seq = []
    visibility_seq = []
    valid_seq = []
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe requires strictly increasing timestamps in ms.
        ts_ms = frame_idx + 1
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            vis = np.array([getattr(lm, "visibility", 0.0) or 0.0 for lm in lms],
                           dtype=np.float32)
            landmarks_seq.append(xyz)
            visibility_seq.append(vis)
            valid_seq.append(True)
        else:
            landmarks_seq.append(np.zeros((33, 3), dtype=np.float32))
            visibility_seq.append(np.zeros(33, dtype=np.float32))
            valid_seq.append(False)
        frame_idx += 1

    cap.release()
    if not landmarks_seq:
        return None
    return {
        "landmarks": np.stack(landmarks_seq),
        "visibility": np.stack(visibility_seq),
        "valid": np.array(valid_seq, dtype=bool),
        "fps": float(fps),
    }


def main():
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"Missing pose model at {args.model}")
    if not args.videos.exists():
        sys.exit(f"No video dir at {args.videos} — run `python -m ml.download_data` first")

    args.out.mkdir(parents=True, exist_ok=True)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    classes = sorted([d.name for d in args.videos.iterdir() if d.is_dir()])
    print(f"Classes: {classes}")
    stats = {c: {"ok": 0, "rejected": 0, "skipped": 0} for c in classes}

    for cls in classes:
        cls_in = args.videos / cls
        cls_out = args.out / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        clips = sorted(cls_in.glob("*.avi")) + sorted(cls_in.glob("*.mp4"))

        for vid in tqdm(clips, desc=cls, unit="clip"):
            out_path = cls_out / (vid.stem + ".npz")
            if out_path.exists() and out_path.stat().st_size > 0:
                stats[cls]["skipped"] += 1
                continue
            # Create a fresh landmarker per clip so the internal VIDEO-mode
            # timestamp tracker resets (it requires strictly increasing ts).
            with PoseLandmarker.create_from_options(options) as landmarker:
                result = extract_clip(landmarker, vid, args.max_frames)
            if result is None:
                stats[cls]["rejected"] += 1
                continue
            detect_rate = float(result["valid"].mean()) if len(result["valid"]) else 0.0
            if detect_rate < args.min_detect_rate:
                stats[cls]["rejected"] += 1
                continue
            np.savez_compressed(out_path, **result)
            stats[cls]["ok"] += 1

    print("\nSummary:")
    for cls, s in stats.items():
        print(f"  {cls}: ok={s['ok']}  rejected={s['rejected']}  skipped(existing)={s['skipped']}")


if __name__ == "__main__":
    main()
