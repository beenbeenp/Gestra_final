"""Print a confusion matrix for the trained ActionLSTM on the val split."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from ml.dataset import (
    PoseSequenceDataset,
    split_by_clip,
    ACTION_NAMES,
    NUM_CLASSES,
    FEAT_DIM,
)
from ml.model import ActionLSTM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", type=Path, default=Path("data/poses"))
    p.add_argument("--model", type=Path, default=Path("motion/models/action_lstm.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    all_clips = []
    for cls_dir in sorted(args.poses.iterdir()):
        if cls_dir.is_dir():
            all_clips.extend(sorted(cls_dir.glob("*.npz")))
    _, val_clips = split_by_clip(all_clips, args.val_frac, args.seed)
    val_ds = PoseSequenceDataset(args.poses, val_clips, args.window, args.stride)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ActionLSTM(input_dim=FEAT_DIM, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=ACTION_NAMES))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
