# Vendored from hustvl/Moebius utils_infer.py (see NOTICE).
# Changes vs upstream: removed the `accelerate` import + parameter (it was only
# used to swap the print function) — inference needs neither.
from typing import Dict, List

import torch
from diffusers.models import AutoencoderKL


@torch.no_grad
def encode_clean_latents(
    batch: Dict,
    vae: AutoencoderKL,
    weight_dtype: str = None) -> List[torch.Tensor]:
    if weight_dtype is None:
        weight_dtype = vae.dtype

    latents = vae.encode(batch["images"].to(vae.dtype)).latent_dist.sample().to(weight_dtype)

    masked_image_latents = vae.encode(batch["masked_images"].to(dtype=vae.dtype)).latent_dist.sample().to(weight_dtype)

    # If a NaN is included, warn and replace
    if torch.any(torch.isnan(latents)):
        print("NaN found in latents, replacing with zeros")
        latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
    if torch.any(torch.isnan(masked_image_latents)):
        print("NaN found in masked_image_latents, replacing with zeros")
        masked_image_latents = torch.where(torch.isnan(masked_image_latents), torch.zeros_like(masked_image_latents), masked_image_latents)

    latents = latents * vae.config.scaling_factor
    masked_image_latents = masked_image_latents * vae.config.scaling_factor

    return latents, masked_image_latents


def predict_noise(
    diff_model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    resized_masks: torch.Tensor,
    masked_latents: torch.Tensor,
    timesteps: torch.Tensor,
    input_ids: torch.Tensor,
    guidance_scale: float = 1.0,
    un_cond_input_ids=None) -> torch.Tensor:

    noisy_latents = torch.cat([noisy_latents] * 2)
    resized_masks = torch.cat([resized_masks] * 2)
    masked_latents = torch.cat([masked_latents] * 2)

    assert input_ids.shape[0] % 2 == 0

    latent_model_input = torch.cat([
        noisy_latents, resized_masks, masked_latents], dim=1)

    # Predict the noise residual
    noise_pred = diff_model(
        latent_model_input,
        timesteps=timesteps,
        input_ids=input_ids
    ).sample

    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred_cfg = noise_pred_uncond + \
        guidance_scale * (noise_pred_cond - noise_pred_uncond)

    return noise_pred_cfg
