"""Train the ActionLSTM on extracted pose sequences.

Usage:
    python -m ml.train [--epochs 30] [--batch 32] [--lr 1e-3]

Saves:
    motion/models/action_lstm.pt   — best-val-accuracy checkpoint
    ml/train.log                   — per-epoch metrics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    p.add_argument("--out", type=Path, default=Path("motion/models/action_lstm.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--log", type=Path, default=Path("ml/train.log"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_clips = []
    for cls_dir in sorted(args.poses.iterdir()):
        if cls_dir.is_dir():
            all_clips.extend(sorted(cls_dir.glob("*.npz")))
    if not all_clips:
        sys.exit(f"No .npz files found under {args.poses}")

    train_clips, val_clips = split_by_clip(all_clips, args.val_frac, args.seed)
    print(f"Clips: {len(all_clips)} total, {len(train_clips)} train, {len(val_clips)} val")

    train_ds = PoseSequenceDataset(args.poses, train_clips, args.window, args.stride)
    val_ds = PoseSequenceDataset(args.poses, val_clips, args.window, args.stride)
    print(f"Windows: {len(train_ds)} train, {len(val_ds)} val")

    dist = train_ds.class_distribution()
    print(f"Train class distribution: {dict(zip(ACTION_NAMES, dist.tolist()))}")
    weights = 1.0 / np.maximum(dist.astype(np.float32), 1.0)
    weights /= weights.sum()
    class_weights = torch.from_numpy(weights).float()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0)

    device = "cpu"
    model = ActionLSTM(input_dim=FEAT_DIM, num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    log_entries = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss / max(train_total, 1), 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss / max(val_total, 1), 4),
            "val_acc": round(val_acc, 4),
        }
        log_entries.append(entry)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            marker = " *"
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={entry['train_loss']:.4f} train_acc={entry['train_acc']:.4f} "
              f"val_loss={entry['val_loss']:.4f} val_acc={entry['val_acc']:.4f}{marker}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log_entries, indent=2))
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {args.out}")
    print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
