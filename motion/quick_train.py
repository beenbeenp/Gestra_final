"""Fast in-game training: retrain the personal model with weighted sampling
so the newest recording has higher influence.

Called from main.py after quick_record completes. Trains for 40 epochs
(~15-20s on Apple Silicon) and overwrites action_personal.pt.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.model import ActionTCN
from ml.train_personal import (
    PersonalDataset, ACTION_NAMES, NUM_CLASSES, FEAT_DIM,
    UPPER_BODY_JOINTS, NUM_JOINTS,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

EPOCHS = 40
LR = 1e-3
BATCH = 16
NEW_DATA_WEIGHT = 3.0


def quick_train(newest_dir=None):
    """Train the personal model, weighting newest_dir samples 3x.

    Args:
        newest_dir: Path to the most recent quick recording directory.
                    Samples from this dir get higher weight during training.

    Returns:
        best validation accuracy (float), or 0.0 on failure.
    """
    data_root = _PROJECT_ROOT / "data" / "personal"
    model_out = _PROJECT_ROOT / "motion" / "models" / "action_personal.pt"

    ds = PersonalDataset(data_root, window=20, stride=3)
    if len(ds) == 0:
        print("quick_train: no data found", file=sys.stderr)
        return 0.0

    # Build per-sample weights: entries from newest_dir get NEW_DATA_WEIGHT
    sample_weights = np.ones(len(ds), dtype=np.float64)
    if newest_dir is not None:
        newest_dir = Path(newest_dir).resolve()
        # PersonalDataset._scan_dir adds entries in order; the newest dir's
        # entries are at the end. We figure out how many by counting.
        newest_ds = PersonalDataset(newest_dir, window=20, stride=3)
        n_new = len(newest_ds)
        if n_new > 0 and n_new <= len(ds):
            sample_weights[-n_new:] = NEW_DATA_WEIGHT
            print(f"  Weighting {n_new} new windows at {NEW_DATA_WEIGHT}x")

    # Split
    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Weighted sampler for training set
    train_weights = sample_weights[train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds),
                                     replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # Class weights for loss
    labels = [ds.entries[i][1] for i in range(len(ds))]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    loss_weights = 1.0 / np.maximum(counts, 1.0)
    loss_weights /= loss_weights.sum()

    model = ActionTCN(input_dim=FEAT_DIM, num_classes=NUM_CLASSES,
                      channels=(64, 64), kernel_size=5, dropout=0.3)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(loss_weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"  Training: {len(ds)} windows ({n_train} train, {n_val} val), {EPOCHS} epochs")

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)
        val_acc = correct / max(total, 1)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "num_classes": NUM_CLASSES,
                        "action_names": ACTION_NAMES}, model_out)

        if epoch % 10 == 0:
            print(f"  [{epoch}/{EPOCHS}] val_acc={val_acc:.3f} (best={best_val_acc:.3f})")

    print(f"  Training complete. Best val accuracy: {best_val_acc:.3f}")
    return best_val_acc
