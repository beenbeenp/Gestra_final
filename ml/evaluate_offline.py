"""Offline evaluation: compare rule-based detector vs TCN on recorded personal data.

Extracts the geometric detection logic from upper_body_detector.py into a pure
function that processes recorded npz landmark frames (no webcam needed).

Usage:
    .venv/bin/python -m ml.evaluate_offline
    .venv/bin/python -m ml.evaluate_offline --csv-log results/experiment_log.csv
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from ml.model import ActionTCN
from ml.train_personal import (
    PersonalDataset, ACTION_NAMES, NUM_CLASSES, FEAT_DIM, WINDOW,
    UPPER_BODY_JOINTS, NUM_JOINTS,
    normalize_seq, add_vel_acc,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16

PUNCH_RAISE_THRESH = 0.0
PUNCH_SPEED_THRESH = 0.04
TILT_THRESH = 0.08
COOLDOWN_FRAMES = 12


def rule_based_predict_sequence(landmarks_seq):
    """Apply rule-based detection to a (T, 33, 3) landmark sequence.

    Returns a single predicted action name for the sequence, using the same
    logic as UpperBodyDetector but on pre-recorded data.
    """
    T = landmarks_seq.shape[0]
    prev_wrists = None
    sh_x_buf = collections.deque(maxlen=10)
    baseline_sh_x = None
    cooldown = 0
    action_counts = collections.Counter()

    for t in range(T):
        lms = landmarks_seq[t]
        ls = lms[L_SHOULDER, :2]
        rs = lms[R_SHOULDER, :2]
        lw = lms[L_WRIST, :2]
        rw = lms[R_WRIST, :2]

        sh_width = max(np.linalg.norm(ls - rs), 0.01)
        sh_center_x = (ls[0] + rs[0]) * 0.5
        sh_center_y = (ls[1] + rs[1]) * 0.5

        sh_x_buf.append(sh_center_x)
        if baseline_sh_x is None and len(sh_x_buf) >= 8:
            baseline_sh_x = np.mean(list(sh_x_buf))

        l_height = (lw[1] - sh_center_y) / sh_width
        r_height = (rw[1] - sh_center_y) / sh_width

        wrist_speed = 0.0
        if prev_wrists is not None:
            dl = np.linalg.norm(lw - prev_wrists[:2]) / sh_width
            dr = np.linalg.norm(rw - prev_wrists[2:]) / sh_width
            wrist_speed = max(dl, dr)
        prev_wrists = np.concatenate([lw, rw])

        action = "idle"
        if cooldown > 0:
            cooldown -= 1
        else:
            if (l_height < PUNCH_RAISE_THRESH or r_height < PUNCH_RAISE_THRESH) and wrist_speed > PUNCH_SPEED_THRESH:
                if l_height < r_height:
                    action = "lpunch"
                else:
                    action = "rpunch"
                cooldown = COOLDOWN_FRAMES

        if action == "idle" and baseline_sh_x is not None:
            tilt = (sh_center_x - baseline_sh_x) / sh_width
            if tilt > TILT_THRESH:
                action = "forward"
            elif tilt < -TILT_THRESH:
                action = "backward"

        action_counts[action] += 1

    return action_counts.most_common(1)[0][0]


def evaluate_rule_based(dataset):
    """Evaluate rule-based detector on PersonalDataset entries."""
    y_true, y_pred = [], []
    start = time.time()
    for seq, label in dataset.entries:
        pred_name = rule_based_predict_sequence(seq)
        y_true.append(label)
        y_pred.append(ACTION_NAMES.index(pred_name))
    elapsed = time.time() - start
    return np.array(y_true), np.array(y_pred), elapsed


def evaluate_tcn(dataset, model_path):
    """Evaluate TCN model on PersonalDataset entries."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ActionTCN(input_dim=FEAT_DIM, num_classes=checkpoint["num_classes"],
                      channels=(64, 64), kernel_size=5, dropout=0.0)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    start = time.time()
    for seq, label in dataset.entries:
        normed = normalize_seq(seq)
        selected = normed[:, UPPER_BODY_JOINTS, :]
        flat = selected.reshape(len(selected), NUM_JOINTS * 3)
        feat = add_vel_acc(flat)
        tensor = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        y_true.append(label)
        y_pred.append(pred)
    elapsed = time.time() - start
    return np.array(y_true), np.array(y_pred), elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=_PROJECT_ROOT / "data" / "personal")
    p.add_argument("--model", type=Path,
                   default=_PROJECT_ROOT / "motion" / "models" / "action_personal.pt")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--csv-log", type=Path, default=None)
    args = p.parse_args()

    ds = PersonalDataset(args.data, window=args.window)
    if len(ds) == 0:
        sys.exit("No data found.")

    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    _, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    val_entries = [ds.entries[i] for i in val_ds.indices]

    class ValSubset:
        def __init__(self, entries):
            self.entries = entries
    val_subset = ValSubset(val_entries)

    print(f"Evaluating on {len(val_entries)} val windows\n")

    print("=" * 50)
    print("RULE-BASED DETECTOR")
    print("=" * 50)
    y_true_r, y_pred_r, time_r = evaluate_rule_based(val_subset)
    acc_r = (y_true_r == y_pred_r).mean()
    print(f"Accuracy: {acc_r:.4f}")
    print(f"Time: {time_r:.3f}s ({time_r / len(val_entries) * 1000:.1f} ms/window)")
    print(classification_report(y_true_r, y_pred_r, target_names=ACTION_NAMES,
                                zero_division=0))

    print("=" * 50)
    print("TCN MODEL")
    print("=" * 50)
    y_true_t, y_pred_t, time_t = evaluate_tcn(val_subset, args.model)
    acc_t = (y_true_t == y_pred_t).mean()
    print(f"Accuracy: {acc_t:.4f}")
    print(f"Time: {time_t:.3f}s ({time_t / len(val_entries) * 1000:.1f} ms/window)")
    print(classification_report(y_true_t, y_pred_t, target_names=ACTION_NAMES,
                                zero_division=0))

    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Rule-based accuracy: {acc_r:.4f}")
    print(f"TCN accuracy:        {acc_t:.4f}")
    print(f"Improvement:         {acc_t - acc_r:+.4f}")

    if args.csv_log:
        args.csv_log.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv_log.exists()
        with open(args.csv_log, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["run_id", "model", "optimizer", "learning_rate",
                            "batch_size", "weight_decay", "window_size", "epochs",
                            "train_accuracy", "validation_accuracy", "training_time_s",
                            "notes"])
            w.writerow(["RULE", "RuleBased", "n/a", "n/a", "n/a", "n/a",
                         args.window, "n/a", "n/a", f"{acc_r:.4f}",
                         f"{time_r:.1f}", "offline eval on val split"])
            w.writerow(["TCN-EVAL", "ActionTCN", "n/a", "n/a", "n/a", "n/a",
                         args.window, "n/a", "n/a", f"{acc_t:.4f}",
                         f"{time_t:.1f}", "offline eval on val split"])
        print(f"\nResults appended to {args.csv_log}")


if __name__ == "__main__":
    main()
