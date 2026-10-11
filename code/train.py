"""Shared training loop for all tasks.

Key design decisions (from roadmap v4):
- AMP on CUDA (fp16 forward, fp32 loss/grad); graceful no-op on MPS/CPU
- SGD, momentum=0.9, wd=5e-4, peak LR=0.1
- LR: linear warmup per-iter epochs 1-5, then CosineAnnealing T_max=55 epochs 6-60
- All 60 epochs always run (early stop is done post-hoc from saved logits)
- Per-epoch test+val logits saved only when save_logits=True (run 2 / Task 2)
- Final checkpoint always saved
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import augment as aug_mod


# ── Device-aware AMP helpers ─────────────────────────────────────────────────
# torch.cuda.amp.GradScaler and autocast work on CUDA only.
# On MPS or CPU we use no-op equivalents so the training loop is identical.

def _make_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    # No-op scaler for MPS / CPU
    class _NoOpScaler:
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass
    return _NoOpScaler()


def _autocast(device: torch.device):
    """Return an autocast context manager appropriate for the device."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()   # MPS / CPU: no autocast


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), targets)


def sce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 1.0,
    A: float = -4.0,
) -> torch.Tensor:
    """Symmetric Cross-Entropy Loss (Wang et al., ICCV 2019).

    L = alpha * H(y, p) + beta * H(p, y)

    The reverse term H(p, y) = -sum_k p_k * log(y_k).
    y (one-hot) is clamped to exp(A) to avoid log(0).
    p is NOT clamped — clamp is on y, not p.
    logits cast to fp32 to keep loss stable under AMP.
    """
    logits = logits.float()
    # Forward CE: H(y, p)
    forward = F.cross_entropy(logits, targets)
    # Reverse CE: H(p, y)
    p = F.softmax(logits, dim=1)
    y = F.one_hot(targets, num_classes=logits.shape[1]).float()
    y = y.clamp(min=math.exp(A))                          # clamp y, NOT p
    reverse = -(p * torch.log(y)).sum(dim=1).mean()
    return alpha * forward + beta * reverse


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, total_epochs: int = 60, warmup_epochs: int = 5):
    """Linear warmup (epochs 1–warmup_epochs) then cosine over remaining epochs."""
    warmup_iters_placeholder = None   # actual steps set in train_one_epoch

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=0.0,
    )
    return cosine


def warmup_lr(optimizer, step: int, warmup_steps: int, peak_lr: float):
    """Set LR for the current global step during warm-up (called per iteration)."""
    lr = peak_lr * step / warmup_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_images_padded: torch.Tensor,   # [N_train, C, 40, 40] float32 GPU — padded+normalized
    train_labels: np.ndarray,            # [N_train] int — labels in train-split order
    val_images: torch.Tensor,            # [N_val, C, 32, 32] float32 GPU — normalized, no pad
    val_noisy_labels: np.ndarray,        # [N_val] int
    val_clean_labels: np.ndarray | None, # [N_val] int — None for tasks 3/4/5
    test_images: torch.Tensor,           # [N_test, C, 32, 32] float32 GPU
    device: torch.device,
    *,
    run_name: str,
    out_dir: str,
    loss_fn=None,                        # defaults to ce_loss
    total_epochs: int = 60,
    warmup_epochs: int = 5,
    batch_size: int = 128,
    peak_lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    save_logits: bool = False,           # save per-epoch test+val logits (only run 2)
    # For co-teaching: train_labels_peer / model_peer passed externally
    peer_model: nn.Module | None = None,
    noise_rate_coteach: float = 0.0,
    # Dynamics logging (task 2)
    mislabeled_mask: np.ndarray | None = None,   # bool [N_train], True = noisy≠clean
    log_dynamics: bool = False,
) -> dict:
    """Train model for total_epochs. Returns a results dict with logged metrics."""
    os.makedirs(out_dir, exist_ok=True)
    if loss_fn is None:
        loss_fn = ce_loss

    N_train = train_images_padded.shape[0]
    N_val   = val_images.shape[0]
    N_test  = test_images.shape[0]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0,   # LR set by scheduler/warmup
        momentum=momentum, weight_decay=weight_decay, nesterov=True
    )
    cosine_sched = make_scheduler(optimizer, total_epochs, warmup_epochs)
    scaler = _make_scaler(device)

    steps_per_epoch = math.ceil(N_train / batch_size)
    warmup_steps    = warmup_epochs * steps_per_epoch
    global_step     = 0

    if peer_model is not None:
        peer_optimizer = torch.optim.SGD(
            peer_model.parameters(), lr=0.0,
            momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
        peer_cosine = make_scheduler(peer_optimizer, total_epochs, warmup_epochs)
        peer_scaler = _make_scaler(device)

    # Pre-allocate logit storage (only if needed)
    if save_logits:
        all_test_logits = np.zeros((total_epochs, N_test, 10), dtype=np.float32)
        all_val_logits  = np.zeros((total_epochs, N_val,  10), dtype=np.float32)

    dynamics_rows = []   # for t2_dynamics.csv

    # Permutation index for shuffling
    idx = np.arange(N_train, dtype=np.int64)

    run_log = {
        "run_name": run_name,
        "epochs": [],
    }

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        model.train()
        if peer_model is not None:
            peer_model.train()

        # Shuffle
        np.random.shuffle(idx)

        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N_train, batch_size):
            batch_idx = idx[start: start + batch_size]
            imgs_padded = train_images_padded[batch_idx]      # [B, C, 40, 40]
            imgs = aug_mod.augment(imgs_padded)               # [B, C, 32, 32]
            lbls = torch.from_numpy(train_labels[batch_idx]).to(device)

            global_step += 1
            # Warm-up LR (per-iteration, epochs 1–warmup_epochs)
            if global_step <= warmup_steps:
                warmup_lr(optimizer, global_step, warmup_steps, peak_lr)
                if peer_model is not None:
                    warmup_lr(peer_optimizer, global_step, warmup_steps, peak_lr)

            if peer_model is not None:
                # ── Co-Teaching step ──────────────────────────────────────
                B_actual = imgs.shape[0]
                keep = max(1, min(B_actual, int(B_actual * _coteach_keep_rate(epoch, noise_rate_coteach))))

                with _autocast(device):
                    logits1 = model(imgs)
                    logits2 = peer_model(imgs)

                # f1 selects keep lowest-loss samples → train f2
                loss1_per = F.cross_entropy(logits1.float(), lbls, reduction="none")
                loss2_per = F.cross_entropy(logits2.float(), lbls, reduction="none")

                sel1 = loss1_per.topk(keep, largest=False).indices
                sel2 = loss2_per.topk(keep, largest=False).indices

                # f2 trains on f1's selection
                with _autocast(device):
                    loss_f2 = F.cross_entropy(logits2[sel1].float(), lbls[sel1])
                peer_optimizer.zero_grad(set_to_none=True)
                peer_scaler.scale(loss_f2).backward()
                peer_scaler.step(peer_optimizer)
                peer_scaler.update()

                # f1 trains on f2's selection
                with _autocast(device):
                    loss_f1 = F.cross_entropy(logits1[sel2].float(), lbls[sel2])
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_f1).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += (loss_f1.item() + loss_f2.item()) / 2

            else:
                # ── Standard / SCE step ──────────────────────────────────
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits = model(imgs)
                loss = loss_fn(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            n_batches += 1

        # Step cosine scheduler (only after warm-up)
        if epoch > warmup_epochs:
            cosine_sched.step()
            if peer_model is not None:
                peer_cosine.step()

        # ── Evaluation ────────────────────────────────────────────────────
        model.eval()
        if peer_model is not None:
            peer_model.eval()

        with torch.no_grad():
            val_logits_ep  = _batch_inference(model, val_images,  device)
            test_logits_ep = _batch_inference(model, test_images, device)
            if peer_model is not None:
                val_logits_ep  = (val_logits_ep  + _batch_inference(peer_model, val_images,  device)) / 2
                test_logits_ep = (test_logits_ep + _batch_inference(peer_model, test_images, device)) / 2

        # Save per-epoch logits (only if requested — task 2)
        if save_logits:
            all_test_logits[epoch - 1] = test_logits_ep
            all_val_logits[epoch - 1]  = val_logits_ep

        # Val accuracy (noisy labels)
        val_pred       = val_logits_ep.argmax(-1)
        val_acc_noisy  = float((val_pred == val_noisy_labels).mean() * 100)
        val_acc_clean  = None
        if val_clean_labels is not None:
            val_acc_clean = float((val_pred == val_clean_labels).mean() * 100)

        # Dynamics logging (task 2)
        if log_dynamics and mislabeled_mask is not None:
            # _batch_inference returns numpy → keep everything in numpy here
            train_logits_ep = _batch_inference(model, train_images_padded[:, :, 4:36, 4:36], device)
            train_pred      = train_logits_ep.argmax(-1)          # numpy [N_train]
            correct_mask    = ~mislabeled_mask                    # numpy bool [N_train]
            mis_mask        =  mislabeled_mask                    # numpy bool [N_train]
            acc_correct    = float((train_pred[correct_mask] == train_labels[correct_mask]).mean() * 100)
            acc_mislabeled = float((train_pred[mis_mask]     == train_labels[mis_mask]    ).mean() * 100)
            val_clean_val  = val_acc_clean if val_acc_clean is not None else 0.0
            dynamics_rows.append([epoch, acc_correct, acc_mislabeled, val_acc_noisy, val_clean_val])

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{run_name}] Epoch {epoch:3d}/{total_epochs} | "
            f"loss={total_loss/n_batches:.4f} | val_noisy={val_acc_noisy:.2f}% | "
            f"lr={current_lr:.5f} | {elapsed:.1f}s"
        )

        ep_log = {
            "epoch": epoch,
            "val_acc_noisy": val_acc_noisy,
            "train_loss": total_loss / n_batches,
            "lr": current_lr,
        }
        if val_acc_clean is not None:
            ep_log["val_acc_clean"] = val_acc_clean
        run_log["epochs"].append(ep_log)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Final checkpoint
    ckpt = {"model": model.state_dict()}
    if peer_model is not None:
        ckpt["peer_model"] = peer_model.state_dict()
    torch.save(ckpt, os.path.join(out_dir, f"{run_name}_final.pt"))

    # Final test predictions CSV
    test_pred_final = test_logits_ep.argmax(-1)
    _save_pred_csv(test_pred_final, os.path.join(out_dir, f"{run_name}_pred.csv"))

    # Per-epoch logits (task 2 only)
    if save_logits:
        np.save(os.path.join(out_dir, f"{run_name}_test_logits.npy"),  all_test_logits)
        np.save(os.path.join(out_dir, f"{run_name}_val_logits.npy"),   all_val_logits)

    # Dynamics CSV (task 2)
    if log_dynamics and dynamics_rows:
        dyn_path = os.path.join(out_dir, f"{run_name}_dynamics.csv")
        _save_dynamics_csv(dynamics_rows, dyn_path)
        run_log["dynamics_path"] = dyn_path

    # Run log JSON
    run_log["final_val_acc_noisy"] = run_log["epochs"][-1]["val_acc_noisy"]
    if "val_acc_clean" in run_log["epochs"][-1]:
        run_log["final_val_acc_clean"] = run_log["epochs"][-1]["val_acc_clean"]
    log_path = os.path.join(out_dir, f"{run_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    return run_log


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_inference(model: nn.Module, images: torch.Tensor, device: torch.device,
                     batch_size: int = 512) -> np.ndarray:
    """Run model in eval mode in batches, return logits [N, 10] as float32 numpy."""
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size].to(device)
            with _autocast(device):
                logits = model(batch)
            all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def _save_pred_csv(preds: np.ndarray, path: str) -> None:
    """Save id,pred CSV. preds: int array [N_test]."""
    with open(path, "w") as f:
        f.write("id,pred\n")
        for i, p in enumerate(preds):
            f.write(f"{i},{int(p)}\n")


def _save_dynamics_csv(rows: list[list], path: str) -> None:
    """Save t2_dynamics.csv with exactly 60 rows and 5 columns, scale 0-100."""
    header = "epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean"
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")


def _coteach_keep_rate(epoch: int, noise_rate: float, T_k: int = 10) -> float:
    """Fraction of mini-batch to keep for co-teaching (gradual warm-up)."""
    tau = min(noise_rate * (epoch / T_k), noise_rate)
    return max(1.0 - tau, 1.0 - noise_rate)   # floor at (1 - noise_rate)


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # MPS: seed is set via torch.manual_seed above (no separate MPS seed API needed)
