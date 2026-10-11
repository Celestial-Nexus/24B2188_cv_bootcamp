"""make_results.py: Read all JSON run logs and generate the filled results.json.

Run AFTER all tasks complete. Reads:
  logs/t1_clean_ref_log.json
  logs/t1_noisy_ce_log.json
  logs/t2_analysis.json
  logs/t4_best.json
  logs/t5_results.json

Writes: results.json (in submission root)

All numeric values in PERCENTAGE scale (0-100).
"""
import json, os, sys

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


def load(name: str) -> dict:
    path = os.path.join(LOG_DIR, name)
    with open(path) as f:
        return json.load(f)


def main():
    t1_clean   = load("t1_clean_ref_log.json")
    t1_noisy   = load("t1_noisy_ce_log.json")
    t2         = load("t2_analysis.json")
    t4         = load("t4_best.json")
    t5         = load("t5_results.json")

    # noise_rate from confusion matrix
    conf = __import__("numpy").load(os.path.join(ROOT_DIR, "code", "confusion_matrix.npy"))
    total_train = conf.sum()
    correct = conf.diagonal().sum()
    noise_rate = float((total_train - correct) / total_train * 100)

    results = {
        "name": "",          # FILL IN
        "roll_no": "",       # FILL IN
        "seed": 42,
        "training_setup": {
            "optimizer": "SGD momentum=0.9 weight_decay=5e-4 nesterov=True",
            "learning_rate_schedule": "linear warmup per-iter epochs 1-5 (0->0.1), cosine T_max=55 epochs 6-60",
            "batch_size": 128,
            "epochs": 60,
            "augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip (GPU-side, per-image offsets)",
            "gpu_used": "",   # FILL IN e.g. "NVIDIA RTX 4090"
        },
        "task1": {
            "noise_rate": round(noise_rate, 2),
            "val_acc_noisy_labels__clean_ref_model":  round(t1_clean["final_val_acc_noisy"], 2),
            "val_acc_noisy_labels__noisy_ce_model":   round(t1_noisy["final_val_acc_noisy"], 2),
        },
        "task2": {
            "memorization_start_epoch": t2["memorization_start_epoch"],
            "early_stop_epoch":         t2["early_stop_epoch"],
            "val_acc_noisy_labels_at_early_stop": round(t2["val_acc_noisy_at_early_stop"], 2),
        },
        "task3": {
            "method": "5-fold stratified CV out-of-fold CE loss (-log p(noisy_label)) "
                      "+ logit-space margin (0.6/0.4 rank combination). "
                      "No class-wise normalization (per-class noise rates span 28-51.5%). "
                      "Signal chosen a priori; AUROC computed post-hoc in code/task3_analysis.json.",
        },
        "task4": {
            "methods": [
                {
                    "name": "sce",
                    "file": "t4_sce.csv",
                    "val_acc_noisy_labels": round(t4["val_accs"]["sce"], 2),
                },
                {
                    "name": "co_teaching",
                    "file": "t4_co_teaching.csv",
                    "val_acc_noisy_labels": round(t4["val_accs"]["co_teaching"], 2),
                },
                {
                    "name": "filter_retrain",
                    "file": "t4_filter_retrain.csv",
                    "val_acc_noisy_labels": round(t4["val_accs"]["filter_retrain"], 2),
                },
            ],
            "best_method": t4["best_method"],
        },
        "task5": {
            "val_acc_synthetic_labels__baseline":    round(t5["val_acc_synthetic_baseline"], 2),
            "val_acc_synthetic_labels__best_method": round(t5["val_acc_synthetic_best"], 2),
        },
    }

    out_path = os.path.join(ROOT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results.json written to {out_path}")
    print("Remember to fill in: name, roll_no, gpu_used")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
