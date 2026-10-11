"""Task 5: Real vs Synthetic Noise comparison.

Retrains two models on synthetic_label column:
  1. CE baseline     → t5_baseline_synth.csv
  2. Best self-contained Task 4 method (SCE or co-teaching) → t5_best_synth.csv

Model selection uses SYNTHETIC val labels only.
clean_label and noisy_label are NEVER used in this file.

The t5_method is read from logs/t4_best.json (written by task4.py).
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import augment as aug_mod
from dataset import load_data
from train import train, set_seed, ce_loss, sce_loss
from model import ResNet18

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")

SEED      = 42
N_CLASSES = 10


def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")

    train_images, test_images_npy, labels_df, train_idx, val_idx = load_data(DATA_DIR)

    synth_train = labels_df.loc[train_idx, "synthetic_label"].to_numpy(dtype=np.int64)
    synth_val   = labels_df.loc[val_idx,   "synthetic_label"].to_numpy(dtype=np.int64)

    # GPU setup
    train_images_padded = aug_mod.setup(train_images[train_idx], device)
    val_images_gpu      = aug_mod.normalize_test(train_images[val_idx], device)
    test_images_gpu     = aug_mod.normalize_test(test_images_npy, device)

    # Read which method Task 4 selected for T5
    with open(os.path.join(LOG_DIR, "t4_best.json")) as f:
        t4_info = json.load(f)
    t5_method = t4_info["t5_method"]
    print(f"Task 5 will use method: {t5_method}")

    # ── Run 12: Synthetic CE baseline ────────────────────────────────────────
    print("\n" + "="*60)
    print("Task 5 — Synthetic CE Baseline")
    print("="*60)
    set_seed(SEED)
    model_base = ResNet18(N_CLASSES).to(device)
    log_base = train(
        model=model_base,
        train_images_padded=train_images_padded,
        train_labels=synth_train,
        val_images=val_images_gpu,
        val_noisy_labels=synth_val,    # model selection on synthetic val labels
        val_clean_labels=None,
        test_images=test_images_gpu,
        device=device,
        run_name="t5_baseline_synth",
        out_dir=LOG_DIR,
        loss_fn=ce_loss,
    )
    shutil.copy(os.path.join(LOG_DIR, "t5_baseline_synth_pred.csv"),
                os.path.join(PRED_DIR, "t5_baseline_synth.csv"))
    val_acc_baseline = log_base["final_val_acc_noisy"]   # val_noisy_labels = synth_val here
    print(f"t5_baseline_synth.csv saved | val_acc_synth={val_acc_baseline:.2f}%")

    # ── Run 13: Synthetic best method ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Task 5 — Synthetic {t5_method}")
    print("="*60)
    set_seed(SEED)

    if t5_method == "sce":
        model_best = ResNet18(N_CLASSES).to(device)
        log_best = train(
            model=model_best,
            train_images_padded=train_images_padded,
            train_labels=synth_train,
            val_images=val_images_gpu,
            val_noisy_labels=synth_val,
            val_clean_labels=None,
            test_images=test_images_gpu,
            device=device,
            run_name="t5_best_synth",
            out_dir=LOG_DIR,
            loss_fn=lambda logits, targets: sce_loss(logits, targets, alpha=0.1, beta=1.0, A=-4.0),
        )
    elif t5_method == "co_teaching":
        model_best1 = ResNet18(N_CLASSES).to(device)
        model_best2 = ResNet18(N_CLASSES).to(device)
        log_best = train(
            model=model_best1,
            train_images_padded=train_images_padded,
            train_labels=synth_train,
            val_images=val_images_gpu,
            val_noisy_labels=synth_val,
            val_clean_labels=None,
            test_images=test_images_gpu,
            device=device,
            run_name="t5_best_synth",
            out_dir=LOG_DIR,
            loss_fn=ce_loss,
            peer_model=model_best2,
            noise_rate_coteach=0.4,
        )
    else:
        raise ValueError(f"Unknown t5_method: {t5_method}. Must be 'sce' or 'co_teaching'.")

    shutil.copy(os.path.join(LOG_DIR, "t5_best_synth_pred.csv"),
                os.path.join(PRED_DIR, "t5_best_synth.csv"))
    val_acc_best = log_best["final_val_acc_noisy"]
    print(f"t5_best_synth.csv saved | val_acc_synth={val_acc_best:.2f}%")

    # Save T5 results
    t5_result = {
        "t5_method": t5_method,
        "val_acc_synthetic_baseline": val_acc_baseline,
        "val_acc_synthetic_best": val_acc_best,
        "note": "val_acc is measured against synthetic_label on val split (41.2% noise rate).",
    }
    with open(os.path.join(LOG_DIR, "t5_results.json"), "w") as f:
        json.dump(t5_result, f, indent=2)

    print("\n=== Task 5 complete ===")
    print(f"  Synthetic baseline ({t5_method} vs CE): {val_acc_best:.2f}% vs {val_acc_baseline:.2f}%")


if __name__ == "__main__":
    main()
