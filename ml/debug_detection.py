"""Debug: shows live detection values with skeleton overlay."""

import os, sys, collections
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gestra_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gestra_cache")

import cv2, mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarksConnections

_ROOT = Path(__file__).resolve().parents[1]
NOSE = 0
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
L_HIP, R_HIP = 23, 24

def main():
    model = _ROOT / "motion" / "models" / "pose_landmarker_lite.task"
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    prev_wrists = None
    lean_buf = collections.deque(maxlen=10)
    baseline_lean = None
    fi = 0

    with PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            fi += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), fi)

            if not result.pose_landmarks:
                cv2.putText(frame, "NO POSE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Debug", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
                continue

            lms = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            pts = [(int(l.x*w), int(l.y*h)) for l in lms]
            for c in PoseLandmarksConnections.POSE_LANDMARKS:
                cv2.line(frame, pts[c.start], pts[c.end], (0,255,0), 2)

            ls = np.array([lms[L_SHOULDER].x, lms[L_SHOULDER].y])
            rs = np.array([lms[R_SHOULDER].x, lms[R_SHOULDER].y])
            lw = np.array([lms[L_WRIST].x, lms[L_WRIST].y])
            rw = np.array([lms[R_WRIST].x, lms[R_WRIST].y])
            sw = max(np.linalg.norm(ls - rs), 0.01)

            l_dist = np.linalg.norm(lw - ls) / sw
            r_dist = np.linalg.norm(rw - rs) / sw

            speed = 0.0
            if prev_wrists is not None:
                dl = np.linalg.norm(lw - prev_wrists[:2]) / sw
                dr = np.linalg.norm(rw - prev_wrists[2:]) / sw
                speed = max(dl, dr)
            prev_wrists = np.concatenate([lw, rw])

            # Lean
            lean_val = 0.0
            lean_diff = 0.0
            lh_vis = getattr(lms[L_HIP], "visibility", 0) or 0
            rh_vis = getattr(lms[R_HIP], "visibility", 0) or 0
            if lh_vis > 0.3 and rh_vis > 0.3:
                hip_cy = (lms[L_HIP].y + lms[R_HIP].y) * 0.5
                lean_val = (hip_cy - lms[NOSE].y) / sw
                lean_buf.append(lean_val)
                if baseline_lean is None and len(lean_buf) >= 8:
                    baseline_lean = np.mean(list(lean_buf))
                if baseline_lean is not None and len(lean_buf) >= 3:
                    lean_diff = np.mean(list(lean_buf)[-3:]) - baseline_lean

            y = 30
            def put(text, color=(255,255,255)):
                nonlocal y
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y += 22

            put(f"L dist: {l_dist:.2f} (>1.2)", (0,255,255) if l_dist > 1.2 else (150,150,150))
            put(f"R dist: {r_dist:.2f} (>1.2)", (0,255,255) if r_dist > 1.2 else (150,150,150))
            put(f"Speed:  {speed:.3f} (>0.08)", (0,255,0) if speed > 0.08 else (150,150,150))
            put(f"Lean:   {lean_diff:+.3f} (|>0.15|)", (0,255,0) if abs(lean_diff) > 0.15 else (150,150,150))
            if baseline_lean: put(f"Baseline: {baseline_lean:.2f}  Now: {lean_val:.2f}")

            action = "idle"
            if l_dist > 1.2 and speed > 0.08: action = "L PUNCH"
            elif r_dist > 1.2 and speed > 0.08: action = "R PUNCH"
            elif lean_diff < -0.15: action = "FORWARD"
            elif lean_diff > 0.15: action = "BACKWARD"
            color = (0,0,255) if action != "idle" else (100,100,100)
            put(f"=> {action}", color)

            cv2.imshow("Debug", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
