"""CIFARNDataset: wraps npy images + labels.csv for CIFAR-10N.

Design principles:
  - clean_label is never loaded in task3/4/5 scripts — enforced by not accepting
    label_column='clean_label' except when task_name='task1_ref'.
  - label_override allows passing an arbitrary label array (for CV folds, filter+retrain).
    Only permitted for task names in OVERRIDE_ALLOWED_TASKS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OVERRIDE_ALLOWED_TASKS = {"task3_cv", "task4_filter"}


class CIFARNDataset(Dataset):
    """Dataset for CIFAR-10N.

    Args:
        images_npy:     numpy array [50000, 32, 32, 3] uint8 (all 50k images).
        indices:        list/array of CIFAR indices (e.g. train_idx or val_idx).
        labels_df:      pandas DataFrame with columns from labels.csv.
        label_column:   one of 'noisy_label', 'synthetic_label', 'clean_label'.
                        'clean_label' is only allowed when task_name='task1_ref'.
        label_override: numpy int array [len(indices)] — bypasses label_column.
                        Only allowed for tasks in OVERRIDE_ALLOWED_TASKS.
        task_name:      used for the label-source guard.
        transform:      callable applied to the raw uint8 image [H, W, C] → tensor.
                        Pass None when images are pre-loaded to GPU and augmented there.
    """

    LABEL_COLUMNS = ("noisy_label", "synthetic_label", "clean_label")

    def __init__(
        self,
        images_npy: np.ndarray,
        indices: list[int] | np.ndarray,
        labels_df: pd.DataFrame,
        label_column: str = "noisy_label",
        label_override: np.ndarray | None = None,
        task_name: str = "unspecified",
        transform=None,
    ):
        self._validate(label_column, label_override, task_name)

        self.indices = np.asarray(indices, dtype=np.int64)
        self.images  = images_npy        # kept as reference, not copied
        self.transform = transform

        if label_override is not None:
            self.labels = np.asarray(label_override, dtype=np.int64)
        else:
            self.labels = labels_df.loc[self.indices, label_column].to_numpy(dtype=np.int64)

    @staticmethod
    def _validate(label_column: str, label_override, task_name: str) -> None:
        if label_column not in CIFARNDataset.LABEL_COLUMNS:
            raise ValueError(f"Unknown label_column '{label_column}'")
        if label_column == "clean_label" and task_name != "task1_ref":
            raise ValueError(
                f"clean_label is only permitted for task_name='task1_ref', got '{task_name}'. "
                "Tasks 2–5 must never train or filter on clean labels."
            )
        if label_override is not None and task_name not in OVERRIDE_ALLOWED_TASKS:
            raise ValueError(
                f"label_override is only permitted for tasks {OVERRIDE_ALLOWED_TASKS}, "
                f"got '{task_name}'."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        cifar_idx = self.indices[pos]
        img = self.images[cifar_idx]   # [32, 32, 3] uint8
        lbl = int(self.labels[pos])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl, pos           # pos = position within this dataset (for OOF scoring)


def load_data(data_dir: str):
    """Load all data files. Returns (train_images_npy, test_images_npy, labels_df,
    train_idx, val_idx)."""
    import os
    train_images = np.load(os.path.join(data_dir, "train_images.npy"), mmap_mode="r")
    test_images  = np.load(os.path.join(data_dir, "test_images.npy"),  mmap_mode="r")
    labels_df    = pd.read_csv(os.path.join(data_dir, "labels.csv"), index_col="index")

    with open(os.path.join(data_dir, "train_idx.txt")) as f:
        train_idx = [int(l) for l in f if l.strip()]
    with open(os.path.join(data_dir, "val_idx.txt")) as f:
        val_idx = [int(l) for l in f if l.strip()]

    return train_images, test_images, labels_df, train_idx, val_idx
