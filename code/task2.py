"""Task 2: Training dynamics analysis and early-stopped predictions.

Reads saved logits from Task 1 (run 2) — no new training needed.

Inputs (from logs/):
  t1_noisy_ce_test_logits.npy  [60, 10000, 10]
  t1_noisy_ce_val_logits.npy   [60, 5000,  10]
  t1_noisy_ce_dynamics.csv     (already copied to predictions/t2_dynamics.csv by task1.py)

Outputs:
  predictions/t2_early_stop.csv
  logs/t2_analysis.json         (memorization_start_epoch, early_stop_epoch, etc.)

Memorization start definition (from roadmap v4):
  smoothed[e] = 3-point centered moving average of train_acc_mislabeled
  baseline    = min(smoothed[0:5])
  first epoch E (1-indexed) such that smoothed[E-1], smoothed[E], smoothed[E+1]
                all exceed baseline + 5.0 pp
  Cross-check: val_acc_clean peaks within ±2 epochs.

Early stop: argmax(val_acc_noisy) over 60 epochs; ties → earlier epoch.

clean_label is loaded here ONLY for the dynamics cross-check (explicitly allowed
by the rules for Task 2 measurement). It is NOT used for model selection.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from dataset import load_data
from train import _save_pred_csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")


def find_memorization_start(dynamics_rows: list[list]) -> int | None:
    """Return 1-indexed epoch where mislabeled accuracy starts rising sustainably.

    Definition:
      smoothed[e] = 3-point centered moving average of train_acc_mislabeled
      baseline = min(smoothed[0:5])
      First epoch E where smoothed[E-1], smoothed[E], smoothed[E+1] all > baseline + 5.0
    """
    mislabeled_acc = np.array([row[2] for row in dynamics_rows])   # 0-indexed
    # 3-point centered moving average (valid mode drops first and last)
    smoothed = np.convolve(mislabeled_acc, np.ones(3) / 3, mode="valid")
    # smoothed[0] corresponds to epoch 2 (1-indexed)
    baseline = float(smoothed[:5].min())
    threshold = baseline + 5.0

    for i in range(len(smoothed) - 2):
        if smoothed[i] > threshold and smoothed[i+1] > threshold and smoothed[i+2] > threshold:
            # smoothed[i] corresponds to epoch i+2 (1-indexed) due to valid conv offset
            return i + 2   # 1-indexed epoch
    return None


def find_early_stop_epoch(val_logits: np.ndarray, val_noisy_labels: np.ndarray) -> tuple[int, float]:
    """Return (1-indexed epoch, val_acc_noisy) using argmax; ties → earlier epoch.

    val_logits: [60, N_val, 10]
    np.argmax returns the FIRST occurrence of the maximum, so ties naturally
    resolve to the earlier epoch — no reversal needed.
    """
    val_preds = val_logits.argmax(-1)   # [60, N_val]
    acc = (val_preds == val_noisy_labels[None, :]).mean(axis=1) * 100   # [60]
    best_0idx = int(np.argmax(acc))     # first occurrence = earliest epoch on tie
    return best_0idx + 1, float(acc[best_0idx])


def main():
    _, _, labels_df, train_idx, val_idx = load_data(DATA_DIR)

    val_noisy = labels_df.loc[val_idx, "noisy_label"].to_numpy(dtype=np.int64)
    val_clean = labels_df.loc[val_idx, "clean_label"].to_numpy(dtype=np.int64)

    # Load saved logits
    test_logits = np.load(os.path.join(LOG_DIR, "t1_noisy_ce_test_logits.npy"))  # [60, 10k, 10]
    val_logits  = np.load(os.path.join(LOG_DIR, "t1_noisy_ce_val_logits.npy"))   # [60, 5k, 10]
    print(f"Loaded logits: test {test_logits.shape}, val {val_logits.shape}")

    # Load dynamics CSV
    dyn_path = os.path.join(PRED_DIR, "t2_dynamics.csv")
    dynamics_rows = []
    with open(dyn_path) as f:
        next(f)   # skip header
        for line in f:
            parts = line.strip().split(",")
            dynamics_rows.append([float(p) for p in parts])
    print(f"Dynamics rows loaded: {len(dynamics_rows)}")

    # ── Memorization start ───────────────────────────────────────────────────
    mem_start = find_memorization_start(dynamics_rows)
    print(f"Memorization start epoch: {mem_start}")

    # Cross-check: val_acc_clean peaks
    val_clean_acc = [row[4] for row in dynamics_rows]
    val_clean_peak_epoch = int(np.argmax(val_clean_acc)) + 1   # 1-indexed
    print(f"val_acc_clean peaks at epoch: {val_clean_peak_epoch}")
    if mem_start is not None:
        diff = abs(mem_start - val_clean_peak_epoch)
        print(f"Cross-check: |memorization_start - val_clean_peak| = {diff} epochs "
              f"({'✓ within 2' if diff <= 2 else '⚠ outside 2'})")

    # ── Early stop ───────────────────────────────────────────────────────────
    early_stop_epoch, early_stop_acc = find_early_stop_epoch(val_logits, val_noisy)
    print(f"Early stop epoch: {early_stop_epoch} | val_acc_noisy={early_stop_acc:.2f}%")

    # Save early-stop predictions
    test_pred_early = test_logits[early_stop_epoch - 1].argmax(-1)   # 0-indexed
    _save_pred_csv(test_pred_early, os.path.join(PRED_DIR, "t2_early_stop.csv"))
    print("t2_early_stop.csv saved")

    # ── Save analysis JSON ────────────────────────────────────────────────────
    analysis = {
        "memorization_start_epoch": mem_start,
        "val_clean_peak_epoch": val_clean_peak_epoch,
        "early_stop_epoch": early_stop_epoch,
        "val_acc_noisy_at_early_stop": early_stop_acc,
        "val_acc_noisy_at_epoch_60": float(
            (val_logits[-1].argmax(-1) == val_noisy).mean() * 100
        ),
    }
    with open(os.path.join(LOG_DIR, "t2_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print("t2_analysis.json saved")
    print("\n=== Task 2 complete ===")
    for k, v in analysis.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
