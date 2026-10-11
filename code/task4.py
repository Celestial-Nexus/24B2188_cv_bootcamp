"""Task 4: Robust training methods.

Methods:
  1. SCE (Symmetric Cross-Entropy)  — t4_sce.csv
  2. Co-Teaching                    — t4_co_teaching.csv
  3. Filter + Retrain               — t4_filter_retrain.csv

All use the same SGD/cosine/augment config as Task 1.
Final epoch (60) predictions for all.
Co-teaching: averaged softmax of both networks → argmax.

Model selection: highest val_acc_noisy among the 3 methods.
Best method written to logs/t4_best.json.

clean_label is NEVER loaded or used in this file.
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import augment as aug_mod
from dataset import load_data
from train import train, set_seed, ce_loss, sce_loss, _batch_inference, _save_pred_csv
from model import ResNet18

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")

SEED        = 42
N_CLASSES   = 10
FILTER_KEEP = 0.60   # keep lowest 60% by noise score


def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")

    train_images, test_images_npy, labels_df, train_idx, val_idx = load_data(DATA_DIR)

    noisy_labels = labels_df.loc[train_idx, "noisy_label"].to_numpy(dtype=np.int64)
    val_noisy    = labels_df.loc[val_idx,   "noisy_label"].to_numpy(dtype=np.int64)

    # GPU setup
    train_images_padded = aug_mod.setup(train_images[train_idx], device)
    val_images_gpu      = aug_mod.normalize_test(train_images[val_idx], device)
    test_images_gpu     = aug_mod.normalize_test(test_images_npy, device)

    val_accs = {}

    # ── Method 1: SCE ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Task 4 — Method 1: Symmetric Cross-Entropy (SCE)")
    print("="*60)
    set_seed(SEED)
    model_sce = ResNet18(N_CLASSES).to(device)
    log_sce = train(
        model=model_sce,
        train_images_padded=train_images_padded,
        train_labels=noisy_labels,
        val_images=val_images_gpu,
        val_noisy_labels=val_noisy,
        val_clean_labels=None,
        test_images=test_images_gpu,
        device=device,
        run_name="t4_sce",
        out_dir=LOG_DIR,
        loss_fn=lambda logits, targets: sce_loss(logits, targets, alpha=0.1, beta=1.0, A=-4.0),
    )
    shutil.copy(os.path.join(LOG_DIR, "t4_sce_pred.csv"),
                os.path.join(PRED_DIR, "t4_sce.csv"))
    val_accs["sce"] = log_sce["final_val_acc_noisy"]
    print(f"t4_sce.csv saved | val_acc_noisy={val_accs['sce']:.2f}%")

    # ── Method 2: Co-Teaching ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Task 4 — Method 2: Co-Teaching (2 networks)")
    print("="*60)
    set_seed(SEED)
    model_ct1 = ResNet18(N_CLASSES).to(device)
    model_ct2 = ResNet18(N_CLASSES).to(device)
    log_ct = train(
        model=model_ct1,
        train_images_padded=train_images_padded,
        train_labels=noisy_labels,
        val_images=val_images_gpu,
        val_noisy_labels=val_noisy,
        val_clean_labels=None,
        test_images=test_images_gpu,
        device=device,
        run_name="t4_co_teaching",
        out_dir=LOG_DIR,
        loss_fn=ce_loss,
        peer_model=model_ct2,
        noise_rate_coteach=0.4,   # from problem statement ("about 40%")
    )
    shutil.copy(os.path.join(LOG_DIR, "t4_co_teaching_pred.csv"),
                os.path.join(PRED_DIR, "t4_co_teaching.csv"))
    val_accs["co_teaching"] = log_ct["final_val_acc_noisy"]
    print(f"t4_co_teaching.csv saved | val_acc_noisy={val_accs['co_teaching']:.2f}%")
    print("(Predictions = averaged softmax of both networks → argmax)")

    # ── Method 3: Filter + Retrain ────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Task 4 — Method 3: Filter + Retrain (keep {int(FILTER_KEEP*100)}% lowest noise score)")
    print("="*60)

    # Load noise scores
    score_path = os.path.join(PRED_DIR, "t3_noise_scores.csv")
    scores = {}
    with open(score_path) as f:
        next(f)
        for line in f:
            idx_str, score_str = line.strip().split(",")
            scores[int(idx_str)] = float(score_str)

    # Sort train_idx by score, keep lowest FILTER_KEEP fraction
    scored = [(scores[cifar_idx], pos) for pos, cifar_idx in enumerate(train_idx)]
    scored.sort()
    keep_n = int(len(scored) * FILTER_KEEP)
    keep_positions = np.array([pos for _, pos in scored[:keep_n]], dtype=np.int64)
    print(f"Keeping {keep_n} / {len(train_idx)} samples")

    filtered_padded = train_images_padded[keep_positions]
    filtered_labels = noisy_labels[keep_positions]

    set_seed(SEED)
    model_fr = ResNet18(N_CLASSES).to(device)
    log_fr = train(
        model=model_fr,
        train_images_padded=filtered_padded,
        train_labels=filtered_labels,
        val_images=val_images_gpu,
        val_noisy_labels=val_noisy,
        val_clean_labels=None,
        test_images=test_images_gpu,
        device=device,
        run_name="t4_filter_retrain",
        out_dir=LOG_DIR,
        loss_fn=ce_loss,
    )
    shutil.copy(os.path.join(LOG_DIR, "t4_filter_retrain_pred.csv"),
                os.path.join(PRED_DIR, "t4_filter_retrain.csv"))
    val_accs["filter_retrain"] = log_fr["final_val_acc_noisy"]
    print(f"t4_filter_retrain.csv saved | val_acc_noisy={val_accs['filter_retrain']:.2f}%")

    # ── Select best method ────────────────────────────────────────────────────
    best_name = max(val_accs, key=val_accs.get)
    print("\n=== Task 4 results ===")
    for name, acc in val_accs.items():
        star = " ← best" if name == best_name else ""
        print(f"  {name:20s}: {acc:.2f}%{star}")

    # Task 5 best method: NEVER filter_retrain (needs new T3 runs on synthetic labels)
    # Use the best self-contained method (SCE or co-teaching)
    self_contained = {k: v for k, v in val_accs.items() if k != "filter_retrain"}
    t5_method = max(self_contained, key=self_contained.get)

    result = {
        "val_accs": val_accs,
        "best_method": best_name,
        "t5_method": t5_method,
        "note_t5": "Task 5 always uses a self-contained method (SCE or co-teaching) "
                   "because filter+retrain would need new Task 3 runs on synthetic labels.",
    }
    with open(os.path.join(LOG_DIR, "t4_best.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTask 4 best method: {best_name}")
    print(f"Task 5 will use   : {t5_method}")


if __name__ == "__main__":
    main()
