"""Task 1: Noise analysis + clean reference model + noisy CE baseline.

Outputs:
  predictions/t1_clean_ref.csv   — final epoch, trained on clean_label
  predictions/t1_noisy_ce.csv    — final epoch, trained on noisy_label
  code/confusion_matrix.npy      — 10x10 C[clean][noisy] on train split
  logs/t1_clean_ref_log.json
  logs/t1_noisy_ce_log.json
  logs/t1_noisy_ce_test_logits.npy  — [60, 10000, 10] for Task 2
  logs/t1_noisy_ce_val_logits.npy   — [60, 5000, 10]  for Task 2
  logs/t1_noisy_ce_dynamics.csv     — for Task 2 (60 rows)

clean_label is only used in this file (task1_ref model).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import augment as aug_mod
from dataset import load_data
from train import train, set_seed, ce_loss, _batch_inference, _save_pred_csv
from model import ResNet18

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs")
PRED_DIR  = os.path.join(os.path.dirname(__file__), "..", "predictions")
CODE_DIR  = os.path.dirname(__file__)

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

SEED = 42


def noise_analysis(labels_df, train_idx: list[int]) -> np.ndarray:
    """Compute and save 10x10 confusion matrix C[clean][noisy] on train split."""
    conf = np.zeros((10, 10), dtype=np.int64)
    train_labels = labels_df.loc[train_idx]
    for _, row in train_labels.iterrows():
        conf[int(row["clean_label"])][int(row["noisy_label"])] += 1
    np.save(os.path.join(CODE_DIR, "confusion_matrix.npy"), conf)
    noise_rate = (train_labels["noisy_label"] != train_labels["clean_label"]).mean() * 100
    print(f"Noise rate: {noise_rate:.2f}%")
    print("Top confusion pairs (true → noisy):")
    pairs = [(conf[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    for cnt, c, n in sorted(pairs, reverse=True)[:10]:
        print(f"  class {c} → class {n}: {cnt}")
    return conf, noise_rate


def main():
    set_seed(SEED)
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")

    train_images, test_images_npy, labels_df, train_idx, val_idx = load_data(DATA_DIR)

    # ── Noise analysis (CPU) ─────────────────────────────────────────────────
    conf, noise_rate = noise_analysis(labels_df, train_idx)
    print(f"Saved confusion_matrix.npy | noise_rate={noise_rate:.2f}%")

    # ── GPU setup ────────────────────────────────────────────────────────────
    train_images_sub = train_images[train_idx]   # [45000, 32, 32, 3] uint8
    train_images_padded = aug_mod.setup(train_images_sub, device)  # [45k, C, 40, 40]
    test_images_gpu  = aug_mod.normalize_test(test_images_npy, device)   # [10k, C, 32, 32]
    val_images_sub   = train_images[val_idx]     # [5000, 32, 32, 3]
    val_images_gpu   = aug_mod.normalize_test(val_images_sub, device)    # [5k, C, 32, 32]

    # Labels
    train_noisy  = labels_df.loc[train_idx, "noisy_label"].to_numpy(dtype=np.int64)
    train_clean  = labels_df.loc[train_idx, "clean_label"].to_numpy(dtype=np.int64)
    val_noisy    = labels_df.loc[val_idx,   "noisy_label"].to_numpy(dtype=np.int64)
    val_clean    = labels_df.loc[val_idx,   "clean_label"].to_numpy(dtype=np.int64)

    # Mislabeled mask for dynamics logging
    mislabeled_mask = (train_noisy != train_clean)   # [45000] bool

    # ── Run 1: Clean Reference ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Run 1: Clean Reference (label_column=clean_label)")
    print("="*60)
    set_seed(SEED)
    model_clean = ResNet18(num_classes=10).to(device)
    log_clean = train(
        model=model_clean,
        train_images_padded=train_images_padded,
        train_labels=train_clean,              # ← clean labels (task1_ref only)
        val_images=val_images_gpu,
        val_noisy_labels=val_noisy,
        val_clean_labels=val_clean,
        test_images=test_images_gpu,
        device=device,
        run_name="t1_clean_ref",
        out_dir=OUT_DIR,
        loss_fn=ce_loss,
        save_logits=False,
        log_dynamics=False,
    )
    # Copy final pred CSV to predictions/
    import shutil
    shutil.copy(os.path.join(OUT_DIR, "t1_clean_ref_pred.csv"),
                os.path.join(PRED_DIR, "t1_clean_ref.csv"))
    print(f"t1_clean_ref.csv saved | val_acc_noisy={log_clean['final_val_acc_noisy']:.2f}%")

    # ── Run 2: Noisy CE Baseline + Task 2 logits ─────────────────────────────
    print("\n" + "="*60)
    print("Run 2: Noisy CE Baseline (label_column=noisy_label) + dynamics")
    print("="*60)
    set_seed(SEED)
    model_noisy = ResNet18(num_classes=10).to(device)
    log_noisy = train(
        model=model_noisy,
        train_images_padded=train_images_padded,
        train_labels=train_noisy,
        val_images=val_images_gpu,
        val_noisy_labels=val_noisy,
        val_clean_labels=val_clean,            # allowed: dynamics logging only
        test_images=test_images_gpu,
        device=device,
        run_name="t1_noisy_ce",
        out_dir=OUT_DIR,
        loss_fn=ce_loss,
        save_logits=True,                      # [60, 10k, 10] and [60, 5k, 10]
        mislabeled_mask=mislabeled_mask,
        log_dynamics=True,
    )
    shutil.copy(os.path.join(OUT_DIR, "t1_noisy_ce_pred.csv"),
                os.path.join(PRED_DIR, "t1_noisy_ce.csv"))
    # Dynamics CSV → predictions/
    shutil.copy(os.path.join(OUT_DIR, "t1_noisy_ce_dynamics.csv"),
                os.path.join(PRED_DIR, "t2_dynamics.csv"))
    print(f"t1_noisy_ce.csv saved | val_acc_noisy={log_noisy['final_val_acc_noisy']:.2f}%")
    print("t2_dynamics.csv saved (60 rows)")

    print("\n=== Task 1 complete ===")
    print(f"noise_rate     : {noise_rate:.2f}%")
    print(f"clean_ref val  : {log_clean['final_val_acc_noisy']:.2f}%  (against noisy val labels)")
    print(f"noisy_ce  val  : {log_noisy['final_val_acc_noisy']:.2f}%  (against noisy val labels)")


if __name__ == "__main__":
    main()
