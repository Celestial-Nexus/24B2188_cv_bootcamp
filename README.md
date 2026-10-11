# CV Bootcamp — Noisy Label Learning on CIFAR-10N

**Rishabh Srivastava | Roll No: 24B2188**

A complete pipeline for training image classifiers under noisy labels on the CIFAR-10N dataset (~40% human annotation noise). Implements noise analysis, memorization dynamics, noise detection, and robust training methods.

---

## Results Summary

| Task | Metric | Result |
|------|--------|--------|
| **Task 1** | Noise rate | 40.24% |
| **Task 1** | Clean reference model val acc | 59.16% |
| **Task 1** | Noisy CE baseline val acc | 50.94% |
| **Task 2** | Memorization start epoch | 38 |
| **Task 2** | Early stop epoch | 35 |
| **Task 2** | Val acc at early stop | 55.72% |
| **Task 3** | Noise detection AUROC (combined) | 0.8919 |
| **Task 4** | Best method (Co-Teaching) val acc | **57.72%** |
| **Task 4** | SCE val acc | 57.24% |
| **Task 4** | Filter+Retrain val acc | 56.50% |
| **Task 5** | Synthetic noise baseline (CE) | 43.74% |
| **Task 5** | Synthetic noise best (Co-Teaching) | **53.52%** |

---

## Repository Structure

```
24B2188_cv_bootcamp/
├── predictions/
│   ├── t1_clean_ref.csv        # Task 1: clean reference model test predictions
│   ├── t1_noisy_ce.csv         # Task 1: noisy CE baseline test predictions
│   ├── t2_dynamics.csv         # Task 2: per-epoch training dynamics (60 rows)
│   ├── t2_early_stop.csv       # Task 2: early-stopped model predictions
│   ├── t3_noise_scores.csv     # Task 3: per-sample noise scores (45k rows)
│   ├── t4_sce.csv              # Task 4: Symmetric Cross-Entropy predictions
│   ├── t4_co_teaching.csv      # Task 4: Co-Teaching predictions (best)
│   ├── t4_filter_retrain.csv   # Task 4: Filter+Retrain predictions
│   ├── t5_baseline_synth.csv   # Task 5: CE baseline on synthetic labels
│   └── t5_best_synth.csv       # Task 5: Co-Teaching on synthetic labels
├── code/
│   ├── README.md               # Detailed reproduction guide
│   ├── task1.py                # Noise analysis + clean ref + noisy CE baseline
│   ├── task2.py                # Memorization dynamics + early stopping
│   ├── task3.py                # 5-fold CV noise detection (OOF CE + logit margin)
│   ├── task4.py                # SCE, Co-Teaching, Filter+Retrain
│   ├── task5.py                # Robust training on synthetic labels
│   ├── train.py                # Core training loop (AMP, co-teaching, dynamics)
│   ├── model.py                # ResNet-18 (~11M params, CIFAR-adapted)
│   ├── augment.py              # GPU-side augmentation (vectorized unfold crop)
│   ├── dataset.py              # CIFARNDataset with clean-label guard
│   └── make_results.py         # Auto-generates results.json from run logs
├── results.json                # All numerical results + training config
└── report.md                   # Q1–Q5 analysis answers
```

---

## Key Design Decisions

- **SCE Loss**: Correct reverse-CE form `-(p * log(y_clamped))` — clamp applied to one-hot label `y`, not to `p`.
- **Task 3 Signal**: `-log p(noisy_label)` OOF loss (AUROC 0.891) combined with logit-space margin `z_best_other - z_noisy` (0.6/0.4 rank blend). No per-class normalization — horse noise (28%) and cat noise (51.5%) differ too much to equalize.
- **Task 5 Self-contained**: Co-Teaching used on synthetic labels without re-running Task 3 (noise scores are derived from `noisy_label`, not `synthetic_label`).
- **GPU Augmentation**: Padding applied *before* normalization (border ≈ -2.0 after norm); crop offsets sampled per-image via vectorized `unfold`, not per-batch.

## Training Setup

- **Architecture**: ResNet-18 (~11.17M params, 3×3 stem, no max-pool)
- **Optimizer**: SGD, momentum=0.9, weight_decay=5e-4, Nesterov=True
- **LR Schedule**: Linear warmup epochs 1–5 (0→0.1), Cosine decay epochs 6–60
- **Batch size**: 128 | **Epochs**: 60 | **Seed**: 42
- **GPU**: NVIDIA Tesla T4 (Google Colab)

## How to Reproduce

See [`code/README.md`](code/README.md) for the full step-by-step reproduction guide.

```bash
# Quick start (from project root, with data/ directory present)
cd code/
python task1.py  # ~18 min on T4
python task2.py  # ~1 min (CPU only)
python task3.py  # ~65 min on T4
python task4.py  # ~40 min on T4
python task5.py  # ~35 min on T4
python make_results.py
```
