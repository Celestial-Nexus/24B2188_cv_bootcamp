"""GPU-side augmentation for CIFAR-10N.

Pipeline (must match torchvision RandomCrop(32, padding=4) + RandomHorizontalFlip):
  raw uint8  →  F.pad(0)  →  normalize  →  random crop per image  →  random flip per image

Padding BEFORE normalization: border value = (0 - mean)/std ≈ -2.0 (black border).
Padding AFTER  normalization: border value = 0.0  (mean-grey) — WRONG, don't do this.
"""
import torch
import torch.nn.functional as F


# ── Normalization constants (computed from 45k train split, NOT val) ─────────
# These are computed once in setup() and cached here.
_MEAN: torch.Tensor | None = None
_STD:  torch.Tensor | None = None


def compute_stats(images_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std from a uint8 image tensor [N, C, H, W]."""
    x = images_uint8.float() / 255.0
    mean = x.mean(dim=[0, 2, 3])   # [C]
    std  = x.std(dim=[0, 2, 3])    # [C]
    return mean, std


def setup(images_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Load all images to GPU, pad, normalize. Returns padded float tensor [N, C, 40, 40].

    Args:
        images_uint8: numpy array [N, H, W, C] uint8 (train split only).
        device: target CUDA device.

    Returns:
        images_padded: float32 tensor [N, C, 40, 40] on device, normalized, zero-padded.
    """
    global _MEAN, _STD
    # HWC → CHW
    t = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).to(device)  # [N, C, 32, 32]
    _MEAN, _STD = compute_stats(t)
    # Pad with 0 BEFORE normalizing: border = (0 - mean)/std ≈ -2.0
    t_padded = F.pad(t.float(), (4, 4, 4, 4), value=0.0)               # [N, C, 40, 40]
    t_padded = (t_padded / 255.0 - _MEAN[:, None, None]) / _STD[:, None, None]
    return t_padded


def augment(imgs: torch.Tensor) -> torch.Tensor:
    """Random crop (32×32) and random horizontal flip, per image.

    Args:
        imgs: float32 tensor [B, C, 40, 40] — padded, normalized batch.

    Returns:
        Augmented tensor [B, C, 32, 32].
    """
    B = imgs.shape[0]
    device = imgs.device
    # Sample independent offsets per image — NOT once per batch (classic bug)
    h_off = torch.randint(0, 9, (B,), device=device)   # 9 = 40 - 32 + 1
    w_off = torch.randint(0, 9, (B,), device=device)
    flip  = torch.rand(B, device=device) > 0.5

    # Vectorized per-image crop using unfold (massively faster on MPS/CUDA)
    unfolded = imgs.unfold(2, 32, 1).unfold(3, 32, 1)  # [B, C, 9, 9, 32, 32]
    b_idx = torch.arange(B, device=device)
    cropped = unfolded[b_idx, :, h_off, w_off]         # [B, C, 32, 32]

    # Per-image horizontal flip
    if flip.any():
        cropped[flip] = cropped[flip].flip(-1)
    return cropped


def normalize_test(images_uint8_npy, device: torch.device) -> torch.Tensor:
    """Normalize test images (no crop, no flip, no padding). [N, C, 32, 32] float32."""
    global _MEAN, _STD
    assert _MEAN is not None, "Call setup() before normalize_test()"
    t = torch.from_numpy(images_uint8_npy).permute(0, 3, 1, 2).to(device).float()
    return (t / 255.0 - _MEAN[:, None, None]) / _STD[:, None, None]


def normalize_batch(imgs_uint8: torch.Tensor) -> torch.Tensor:
    """Normalize a batch that's already on device (eval — no aug). [B, C, 32, 32]."""
    global _MEAN, _STD
    assert _MEAN is not None
    return (imgs_uint8.float() / 255.0 - _MEAN[:, None, None]) / _STD[:, None, None]


def get_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (mean, std) tensors."""
    assert _MEAN is not None
    return _MEAN, _STD
