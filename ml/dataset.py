"""PyTorch Dataset over MediaPipe pose .npz files produced by ml.extract_poses.

We map HMDB51 classes onto our 3-class motion-control vocabulary:
    stand            -> idle
    punch            -> punch
    kick, kick_ball  -> kick

Each .npz contains a (T, 33, 3) landmark sequence and a (T,) `valid` mask. We:
  1. Drop frames with no detected pose.
  2. Hip-center to (0, 0, 0) and divide by shoulder width so the input is
     scale + translation invariant. This makes the model robust to where the
     actor stands in frame.
  3. Slide a fixed-length window (default 30 frames, ~1 s at 30 fps) across
     each clip with stride 5, producing many training examples per clip.
  4. Yield (window, class_idx) tensors.

Test-time inference uses the same normalization in motion/pose_predictor.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Mapping from HMDB51 class folder name to our motion-control class index.
HMDB_TO_ACTION = {
    "stand": 0,
    "talk": 0,
    "smile": 0,
    "sit": 0,
    "drink": 0,
    "punch": 1,
    "kick": 2,
    "kick_ball": 2,
}
ACTION_NAMES = ["idle", "punch", "kick"]
NUM_CLASSES = len(ACTION_NAMES)

# MediaPipe pose landmark indices used for normalization.
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

DEFAULT_WINDOW = 30
DEFAULT_STRIDE = 5
FEAT_DIM = 99 * 3  # position + velocity + acceleration


def normalize_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Hip-center + shoulder-width-normalize a (T, 33, 3) clip.

    Per-frame normalization (each frame normalized using its own hips/shoulders)
    so the model sees pose shape, not absolute body position over time.
    """
    out = landmarks.copy()
    hips = (out[:, LEFT_HIP, :] + out[:, RIGHT_HIP, :]) * 0.5  # (T, 3)
    out -= hips[:, None, :]
    shoulders = np.linalg.norm(
        out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=1
    )  # (T,)
    shoulders = np.maximum(shoulders, 1e-6)
    out /= shoulders[:, None, None]
    return out.astype(np.float32)


def add_velocity_features(seq: np.ndarray) -> np.ndarray:
    """Given (T, 99) position features, append velocity and acceleration.

    Returns (T, 297) = [position | velocity | acceleration].
    Velocity at t=0 is zero-padded; acceleration at t=0,1 is zero-padded.
    """
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


def windows_from_clip(landmarks: np.ndarray, valid: np.ndarray,
                      window: int, stride: int) -> List[np.ndarray]:
    """Slide a fixed-length window over the *valid* frames of a clip.

    Frames where MediaPipe failed to detect a pose are dropped before windowing
    so we don't leak zeros into the LSTM. If the surviving sequence is shorter
    than `window`, returns a single window padded by edge-replication.
    """
    if landmarks.shape[0] == 0:
        return []
    keep = valid.astype(bool)
    if not keep.any():
        return []
    seq = landmarks[keep]
    if seq.shape[0] < window:
        pad = np.repeat(seq[-1:], window - seq.shape[0], axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    out = []
    for start in range(0, seq.shape[0] - window + 1, stride):
        out.append(seq[start : start + window])
    return out


class PoseSequenceDataset(Dataset):
    """All windows from all clips, flat-indexed.

    We materialize windows lazily (per clip) to save memory: __getitem__ loads
    the right .npz, slices to the precomputed (clip, start) pair, and returns.
    """

    def __init__(
        self,
        poses_root: Path,
        clip_paths: List[Path] | None = None,
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
    ):
        self.poses_root = Path(poses_root)
        self.window = window
        self.stride = stride

        if clip_paths is None:
            clip_paths = []
            for cls_dir in sorted(self.poses_root.iterdir()):
                if cls_dir.is_dir() and cls_dir.name in HMDB_TO_ACTION:
                    clip_paths.extend(sorted(cls_dir.glob("*.npz")))
        self.clip_paths = clip_paths

        # Precompute (clip_idx, start_frame_in_kept_seq) entries.
        self.entries: List[Tuple[int, int, int]] = []  # (clip_idx, start, label)
        self.lengths: List[int] = []
        for ci, p in enumerate(self.clip_paths):
            label = HMDB_TO_ACTION[p.parent.name]
            with np.load(p) as data:
                valid = data["valid"]
                kept = int(valid.sum())
            n_starts = max(0, kept - window + 1)
            if n_starts <= 0 and kept > 0:
                # short clip: treat as one window with edge padding
                self.entries.append((ci, 0, label))
                self.lengths.append(kept)
                continue
            for s in range(0, n_starts, stride):
                self.entries.append((ci, s, label))
            self.lengths.append(kept)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        clip_idx, start, label = self.entries[idx]
        with np.load(self.clip_paths[clip_idx]) as data:
            lms = data["landmarks"]
            valid = data["valid"]
        seq = lms[valid.astype(bool)]
        if seq.shape[0] < self.window:
            pad = np.repeat(seq[-1:], self.window - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        window = seq[start : start + self.window]
        window = normalize_sequence(window)
        flat = window.reshape(self.window, 33 * 3).astype(np.float32)
        flat = add_velocity_features(flat)  # (T, 297)
        return torch.from_numpy(flat), torch.tensor(label, dtype=torch.long)

    def class_distribution(self):
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, _, label in self.entries:
            counts[label] += 1
        return counts


def split_by_clip(clip_paths: List[Path], val_frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """Per-clip 80/20 split, stratified by class so each split has all classes."""
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for p in clip_paths:
        by_class.setdefault(p.parent.name, []).append(p)
    train, val = [], []
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_frac)))
        train.extend(items[:cut])
        val.extend(items[cut:])
    return train, val
