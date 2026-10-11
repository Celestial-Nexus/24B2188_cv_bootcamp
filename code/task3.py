"""Task 3: Noise score assignment for every training image.

Strategy:
  Primary signal  : 5-fold stratified CV, per-sample CE loss = -log p(noisy_label)
                    computed on held-out fold (TTA: average logits of original + h-flip)
  Secondary signal: logit-space margin = max_{j≠noisy} z_j  -  z_{noisy}
                    (same CV logits, no extra runs)
  Combination     : 0.6 * rank_norm(primary) + 0.4 * rank_norm(margin)

NO class-wise normalization — verified that per-class noise rates span 28%-51.5%;
per-class rank norm would equalize unequal rates and hurt precision@k.

Output:
  predictions/t3_noise_scores.csv  (index,score — 45k rows, index from train_idx.txt)
  logs/t3_oof_logits.npy           [45000, 10] float32 — OOF logits for analysis
  logs/task3_analysis.json         (signal AUROC vs clean labels — post-hoc only)

clean_label is used ONLY in task3_analysis.json (post-hoc AUROC check).
It is never used to select the signal or to determine which samples to flag.
Signal chosen a priori (OOF CE + logit margin). See README.md.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

import augment as aug_mod
from dataset import load_data
from train import train, set_seed, ce_loss, _batch_inference
from model import ResNet18

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")

SEED      = 42
N_FOLDS   = 5
N_CLASSES = 10


def rank_norm(s: np.ndarray) -> np.ndarray:
    """Rank-normalize array to [0, 1]. Handles ties via average rank."""
    r = rankdata(s, method="average")
    return (r - 1) / (len(s) - 1)


def compute_logit_margin(logits: np.ndarray, noisy_labels: np.ndarray) -> np.ndarray:
    """Logit-space margin: max_{j≠y} z_j - z_y.

    Positive = model prefers a different class than the noisy label → likely mislabeled.
    Stays discriminative in float32 (no saturation unlike prob-space margin).
    """
    n = len(logits)
    z_noisy = logits[np.arange(n), noisy_labels]                    # [N]
    logits_masked = logits.copy()
    logits_masked[np.arange(n), noisy_labels] = -1e9
    z_best_other = logits_masked.max(axis=1)                        # [N]
    return z_best_other - z_noisy                                    # [N]


def tta_logits(model, imgs_padded: torch.Tensor, device: torch.device) -> np.ndarray:
    """Inference with TTA: average of original + horizontal flip logits. [N, 10]."""
    orig  = _batch_inference(model, imgs_padded[:, :, 4:36, 4:36], device)   # centre crop
    flipped = _batch_inference(model, imgs_padded[:, :, 4:36, 4:36].flip(-1), device)
    return (orig + flipped) / 2.0


def main():
    set_seed(SEED)
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")

    train_images, _, labels_df, train_idx, val_idx = load_data(DATA_DIR)

    # Labels — clean_label used only for fold stratification (not for training/scoring)
    noisy_labels = labels_df.loc[train_idx, "noisy_label"].to_numpy(dtype=np.int64)
    clean_labels = labels_df.loc[train_idx, "clean_label"].to_numpy(dtype=np.int64)

    # GPU setup
    train_images_sub    = train_images[train_idx]
    train_images_padded = aug_mod.setup(train_images_sub, device)   # [45k, C, 40, 40]

    N_train = len(train_idx)
    oof_logits = np.zeros((N_train, N_CLASSES), dtype=np.float32)

    # ── 5-Fold Stratified CV ─────────────────────────────────────────────────
    # Stratify on clean_label for balanced class distribution per fold.
    # clean_label is used here ONLY for fold definition — the model trains on noisy_label.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (fold_train_pos, fold_val_pos) in enumerate(
        skf.split(np.arange(N_train), clean_labels), start=1
    ):
        print(f"\n{'='*60}")
        print(f"Task 3 — Fold {fold_idx}/{N_FOLDS}  "
              f"(train={len(fold_train_pos)}, val={len(fold_val_pos)})")
        print("="*60)

        fold_train_padded = train_images_padded[fold_train_pos]
        fold_val_padded   = train_images_padded[fold_val_pos]
        fold_train_labels = noisy_labels[fold_train_pos]

        # Dummy val/test for the train() interface (we don't use val_acc for fold models)
        # Use fold-val images normalized without padding for eval
        fold_val_imgs = fold_val_padded[:, :, 4:36, 4:36]   # centre crop, no aug

        set_seed(SEED + fold_idx)
        model = ResNet18(num_classes=N_CLASSES).to(device)

        _ = train(
            model=model,
            train_images_padded=fold_train_padded,
            train_labels=fold_train_labels,
            val_images=fold_val_imgs,
            val_noisy_labels=noisy_labels[fold_val_pos],
            val_clean_labels=None,        # task3_cv — no clean label access
            test_images=fold_val_imgs,    # reuse val as "test" — we need OOF logits
            device=device,
            run_name=f"t3_cv_fold{fold_idx}",
            out_dir=LOG_DIR,
            loss_fn=ce_loss,
            save_logits=False,
        )

        # Get OOF logits via TTA on the held-out fold
        fold_val_oof_logits = tta_logits(model, fold_val_padded, device)
        oof_logits[fold_val_pos] = fold_val_oof_logits
        print(f"Fold {fold_idx} OOF logits computed ({len(fold_val_pos)} samples)")

    # ── Compute scores ────────────────────────────────────────────────────────
    print("\nComputing noise scores...")
    n = N_train
    # Primary: -log p(noisy_label)
    p_noisy = np.exp(oof_logits - oof_logits.max(axis=1, keepdims=True))
    p_noisy /= p_noisy.sum(axis=1, keepdims=True)
    p_noisy_target = p_noisy[np.arange(n), noisy_labels]
    p_noisy_target = np.clip(p_noisy_target, 1e-9, 1.0)    # avoid log(0)
    primary = -np.log(p_noisy_target)

    # Secondary: logit-space margin
    margin = compute_logit_margin(oof_logits, noisy_labels)

    # Combine
    noise_score = 0.6 * rank_norm(primary) + 0.4 * rank_norm(margin)

    # ── Save scores ───────────────────────────────────────────────────────────
    np.save(os.path.join(LOG_DIR, "t3_oof_logits.npy"), oof_logits)

    score_path = os.path.join(PRED_DIR, "t3_noise_scores.csv")
    with open(score_path, "w") as f:
        f.write("index,score\n")
        for pos, cifar_idx in enumerate(train_idx):
            f.write(f"{cifar_idx},{noise_score[pos]:.6f}\n")
    print(f"t3_noise_scores.csv saved ({n} rows)")

    # ── Post-hoc AUROC (analysis only — never used for any training decision) ─
    try:
        from sklearn.metrics import roc_auc_score
        mislabeled_gt = (noisy_labels != clean_labels).astype(int)
        auroc_primary = roc_auc_score(mislabeled_gt, primary)
        auroc_margin  = roc_auc_score(mislabeled_gt, margin)
        auroc_combined = roc_auc_score(mislabeled_gt, noise_score)
        print(f"\nPost-hoc AUROC (clean-label ground truth — analysis only):")
        print(f"  primary (CE loss): {auroc_primary:.4f}")
        print(f"  margin (logit)   : {auroc_margin:.4f}")
        print(f"  combined         : {auroc_combined:.4f}")
        analysis = {
            "note": "AUROC computed post-hoc against clean labels for analysis only. "
                    "Signal was chosen a priori (OOF CE + logit margin).",
            "auroc_primary": auroc_primary,
            "auroc_margin":  auroc_margin,
            "auroc_combined": auroc_combined,
        }
    except ImportError:
        analysis = {"note": "sklearn not available for AUROC"}

    with open(os.path.join(LOG_DIR, "task3_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    print("\n=== Task 3 complete ===")


if __name__ == "__main__":
    main()
