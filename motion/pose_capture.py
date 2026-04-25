import argparse
import json
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/gestra_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gestra_cache")

try:
    import cv2
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
except ImportError as exc:
    missing = exc.name
    print(
        f"Missing dependency: {missing}. Install motion dependencies with "
        "`python -m pip install -r motion/requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def landmark_to_dict(landmark):
    return {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": getattr(landmark, "visibility", None),
        "presence": getattr(landmark, "presence", None),
    }


def draw_pose_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 180, 255), -1)


def write_sample(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "landmark_schema": "mediapipe_pose_33",
        "frame_count": len(frames),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview webcam pose landmarks.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--save-sample",
        type=Path,
        default=None,
        help="Optional JSON file for a short landmark sample.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=90,
        help="Number of detected-pose frames to save when --save-sample is set.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional maximum runtime before the preview closes.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task",
        help="MediaPipe Pose Landmarker .task model path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera}.")

    if not args.model.exists():
        raise SystemExit(
            f"Missing pose model: {args.model}\n"
            "Download pose_landmarker_lite.task into motion/models before running."
        )

    sample_frames = []
    frame_index = 0
    start_time = time.time()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_index * 33)

            status = "no pose"
            if result.pose_landmarks:
                status = "pose detected"
                landmarks = result.pose_landmarks[0]
                draw_pose_landmarks(frame, landmarks)
                if args.save_sample and len(sample_frames) < args.sample_frames:
                    sample_frames.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": time.time(),
                            "landmarks": [
                                landmark_to_dict(landmark)
                                for landmark in landmarks
                            ],
                        }
                    )

            sample_text = ""
            if args.save_sample:
                sample_text = f" | sample {len(sample_frames)}/{args.sample_frames}"

            cv2.putText(
                frame,
                f"{status}{sample_text} | q/esc quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if result.pose_landmarks else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Gestra Pose Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.max_seconds and time.time() - start_time >= args.max_seconds:
                break
            if args.save_sample and len(sample_frames) >= args.sample_frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    if args.save_sample:
        write_sample(args.save_sample, sample_frames)
        print(f"Saved {len(sample_frames)} pose frames to {args.save_sample}")


if __name__ == "__main__":
    main()
