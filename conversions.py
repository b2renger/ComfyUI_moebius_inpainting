"""ComfyUI tensor <-> PIL conversions.

ComfyUI conventions: IMAGE is [B,H,W,C] float32 in 0..1 (channel-last);
MASK is [B,H,W] (or [H,W]) float32 in 0..1 where 1.0 = region to edit.
Moebius expects PIL RGB images and PIL 'L' masks where white = inpaint —
the same polarity, so no inversion happens here.
"""
import numpy as np
import torch
from PIL import Image


def image_batch_to_pil(image: torch.Tensor) -> list:
    """IMAGE [B,H,W,C] 0..1 -> list of PIL RGB images."""
    arr = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a[..., :3]) for a in arr]


def mask_batch_to_pil(mask: torch.Tensor, batch_size: int, size) -> list:
    """MASK [B,H,W] or [H,W] 0..1 -> list of `batch_size` PIL 'L' masks at `size` (W,H).

    A single mask is broadcast across the batch (matches how core inpaint
    nodes treat a 1-mask/N-image pairing). Masks are resized with NEAREST to
    keep them binary.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    arr = (mask.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    pils = [Image.fromarray(a, mode="L") for a in arr]
    while len(pils) < batch_size:
        pils.append(pils[-1])
    pils = [p if p.size == size else p.resize(size, Image.Resampling.NEAREST) for p in pils]
    return pils[:batch_size]


def pil_to_image_batch(pils: list) -> torch.Tensor:
    """List of PIL images -> IMAGE [B,H,W,3] float32 0..1 (all same size)."""
    arrs = [
        torch.from_numpy(np.asarray(p.convert("RGB")).astype(np.float32) / 255.0)
        for p in pils
    ]
    return torch.stack(arrs)
