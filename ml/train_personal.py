"""Train a personal action model from data recorded by ml.record_data.

Usage:
    .venv/bin/python -m ml.train_personal

Reads from data/personal/{idle,lpunch,rpunch,forward,backward}/*.npz
Saves to motion/models/action_personal.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.model import ActionTCN

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTION_MAP = {"idle": 0, "lpunch": 1, "rpunch": 2, "forward": 3, "backward": 4}
ACTION_NAMES = ["idle", "lpunch", "rpunch", "forward", "backward"]
NUM_CLASSES = len(ACTION_NAMES)
WINDOW = 20  # shorter window for personal data (~0.67s at 30fps)
STRIDE = 3
FEAT_DIM = 99 * 3  # pos + vel + acc

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def normalize_seq(landmarks):
    out = landmarks.copy()
    hips = (out[:, LEFT_HIP, :] + out[:, RIGHT_HIP, :]) * 0.5
    out -= hips[:, None, :]
    sw = np.linalg.norm(out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=1)
    sw = np.maximum(sw, 1e-6)
    out /= sw[:, None, None]
    return out.astype(np.float32)


def add_vel_acc(seq):
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


class PersonalDataset(Dataset):
    def __init__(self, data_root, window=WINDOW, stride=STRIDE):
        self.entries = []
        data_root = Path(data_root)
        self._scan_dir(data_root, window, stride)
        # Also scan person subdirs (data/personal/alice/, data/personal/bob/)
        for sub in sorted(data_root.iterdir()):
            if sub.is_dir() and sub.name not in ACTION_MAP:
                self._scan_dir(sub, window, stride)
        if not self.entries:
            print("No training data found!", file=sys.stderr)

    def _scan_dir(self, root, window, stride):
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in ACTION_MAP:
                continue
            label = ACTION_MAP[cls_dir.name]
            for npz_path in sorted(cls_dir.glob("*.npz")):
                with np.load(npz_path) as data:
                    lms = data["landmarks"]
                    valid = data["valid"]
                seq = lms[valid.astype(bool)]
                if len(seq) < window:
                    if len(seq) > 0:
                        pad = np.repeat(seq[-1:], window - len(seq), axis=0)
                        seq = np.concatenate([seq, pad])
                        self.entries.append((seq[:window], label))
                    continue
                for s in range(0, len(seq) - window + 1, stride):
                    self.entries.append((seq[s:s + window], label))

        if not self.entries:
            print("No training data found!", file=sys.stderr)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        seq, label = self.entries[idx]
        normed = normalize_seq(seq)
        flat = normed.reshape(len(normed), 99)
        feat = add_vel_acc(flat)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=_PROJECT_ROOT / "data" / "personal")
    p.add_argument("--out", type=Path, default=_PROJECT_ROOT / "motion" / "models" / "action_personal.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    ds = PersonalDataset(args.data)
    if len(ds) == 0:
        sys.exit("No data. Run `python -m ml.record_data` first.")

    # 80/20 split
    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val],
                                                      generator=torch.Generator().manual_seed(42))
    print(f"Dataset: {len(ds)} windows ({n_train} train, {n_val} val)")

    # Class weights
    labels = [ds.entries[i][1] for i in range(len(ds))]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= weights.sum()
    print(f"Class counts: {dict(zip(ACTION_NAMES, counts.astype(int).tolist()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = ActionTCN(input_dim=FEAT_DIM, num_classes=NUM_CLASSES,
                      channels=(64, 64), kernel_size=5, dropout=0.3)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += x.size(0)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(1)
                v_correct += (preds == y).sum().item()
                v_total += x.size(0)

        t_acc = t_correct / max(t_total, 1)
        v_acc = v_correct / max(v_total, 1)
        marker = ""
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "num_classes": NUM_CLASSES,
                        "action_names": ACTION_NAMES}, args.out)
            marker = " *"

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"[{epoch:02d}/{args.epochs}] train_acc={t_acc:.3f} val_acc={v_acc:.3f}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {args.out}")
    print("\nTo use it, launch the game with:")
    print("  GESTRA_WEBCAM=1 GESTRA_PERSONAL_MODEL=1 .venv/bin/python main.py")


if __name__ == "__main__":
    main()
