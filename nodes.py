"""Moebius inpainting nodes (V1 ComfyUI node API).

Design notes (see CLAUDE.md for the full rationale):
- Heavy imports (moebius_src, diffusers, cv2) happen inside the node
  functions, not at module import, so ComfyUI startup stays fast.
- Moebius takes NO text prompt: conditioning is a fixed learned embedding
  table inside the checkpoint. The inpaint node is pure image+mask.
- Mask dilation and paste-back happen node-side at the ORIGINAL resolution
  (the vendored pipeline is called with paste=False, dilation 0), so with
  `paste` enabled the unmasked pixels of the output are identical to the
  input, whatever the processing resolution was.
"""
import os

import torch
from PIL import Image, ImageFilter
import numpy as np

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .download import CHECKPOINTS, CKPT_FILE, ensure_checkpoint, ensure_vae
from .conversions import image_batch_to_pil, mask_batch_to_pil, pil_to_image_batch

MOEBIUS_FOLDER = "moebius"
DOWNLOAD_PREFIX = "(download) "

_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _base_dir():
    return folder_paths.get_folder_paths(MOEBIUS_FOLDER)[0]


_MODEL_EXTS = (".bin", ".safetensors", ".ckpt", ".pt", ".pth")


def _model_choices():
    local = []
    for f in folder_paths.get_filename_list(MOEBIUS_FOLDER):
        n = f.replace("\\", "/")
        # skip the VAE (loaded implicitly), dot-dirs and huggingface_hub's
        # .cache/ bookkeeping, and non-checkpoint files
        if n.startswith((".", "vae/")) or "/." in n:
            continue
        if not n.lower().endswith(_MODEL_EXTS):
            continue
        local.append(n)
    downloads = [
        DOWNLOAD_PREFIX + name
        for name in CHECKPOINTS
        if f"{name}/{CKPT_FILE}" not in local
    ]
    return local + downloads


class MoebiusModelLoader:
    DESCRIPTION = (
        "Loads a Moebius inpainting checkpoint (0.22B, hustvl) plus the shared "
        "PixelHacker VAE. '(download)' entries fetch the official weights from "
        "HuggingFace into ComfyUI/models/moebius on first use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_choices(), {
                    "tooltip": "Checkpoint under models/moebius, or a one-click HuggingFace download. "
                               "pretrained = general, ft_places2 = scenes, ft_celebahq/ft_ffhq = faces."}),
                "dtype": (list(_DTYPES), {
                    "default": "fp32",
                    "tooltip": "fp32 recommended by the authors (the model is small, fp32 is cheap)."}),
            }
        }

    RETURN_TYPES = ("MOEBIUS_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "Moebius"

    def load(self, model_name, dtype):
        from .moebius_src import (
            MOEBIUS_CONFIG_PATH,
            MoebiusPipeline,
            build_removal_model,
            load_removal_model,
        )
        from diffusers import AutoencoderKL, DDIMScheduler

        base = _base_dir()
        if model_name.startswith(DOWNLOAD_PREFIX):
            ckpt_path = ensure_checkpoint(model_name[len(DOWNLOAD_PREFIX):], base)
        else:
            ckpt_path = folder_paths.get_full_path(MOEBIUS_FOLDER, model_name)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Moebius checkpoint '{model_name}' not found under models/moebius")
        vae_dir = ensure_vae(base)

        device = mm.get_torch_device()
        torch_dtype = _DTYPES[dtype]

        model = build_removal_model(MOEBIUS_CONFIG_PATH, num_embeddings=20)
        load_removal_model(model, ckpt_path, device="cpu")
        # the published VAE is a .bin; opting out of safetensors avoids a
        # noisy "no safetensors found / unsafe serialization" warning
        vae = AutoencoderKL.from_pretrained(vae_dir, use_safetensors=False)
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            num_train_timesteps=1000, clip_sample=False)

        pipe = MoebiusPipeline(model, vae, scheduler, device=device, dtype=torch_dtype)
        return (pipe,)


class MoebiusInpaint:
    DESCRIPTION = (
        "Mask-driven inpainting / object removal with Moebius. No text prompt: "
        "white (1.0) mask regions are removed and refilled from context."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MOEBIUS_PIPE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                                  "tooltip": "DDIM denoising steps (upstream default 20)."}),
                "guidance": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 15.0, "step": 0.1,
                                       "tooltip": "Classifier-free guidance; upstream uses 2.0-2.5."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "image_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64,
                                       "tooltip": "Processing resolution (short side). The model is trained at 512."}),
                "mask_dilate": ("INT", {"default": 0, "min": 0, "max": 128,
                                        "tooltip": "Grow the mask by N pixels (at input resolution) before inpainting."}),
                "paste": ("BOOLEAN", {"default": True,
                                      "tooltip": "Blend the result into the original image so unmasked pixels stay identical."}),
            },
            "optional": {
                "compensate": ("BOOLEAN", {"default": False,
                                           "tooltip": "Upstream brightness-compensated paste (advanced; implies paste)."}),
                "noise_offset": ("FLOAT", {"default": 0.0357, "min": 0.0, "max": 0.2, "step": 0.0001,
                                           "tooltip": "Initial-noise variance offset (upstream default 0.0357)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "Moebius"

    def inpaint(self, pipe, image, mask, steps, guidance, seed, image_size,
                mask_dilate, paste, compensate=False, noise_offset=0.0357):
        batch = image.shape[0]
        h0, w0 = image.shape[1], image.shape[2]

        pil_images = image_batch_to_pil(image)
        pil_masks = mask_batch_to_pil(mask, batch, (w0, h0))

        # Binarize + dilate at the original resolution so the same mask drives
        # both the model and the node-side paste-back.
        pil_masks = [m.point(lambda x: 0 if x < 128 else 255, "L") for m in pil_masks]
        if mask_dilate > 0:
            import cv2
            kernel = np.ones((mask_dilate, mask_dilate), np.uint8)
            pil_masks = [
                Image.fromarray(cv2.dilate(np.array(m), kernel, iterations=1), mode="L")
                for m in pil_masks
            ]

        if all((np.asarray(m) == 0).all() for m in pil_masks):
            print("[Moebius] mask is empty - returning the input unchanged")
            return (image,)

        pbar = comfy.utils.ProgressBar(steps)

        def step_callback(step, total):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(step, total)

        results = pipe(
            pil_images, pil_masks,
            image_size=image_size,
            mask_dilate_kernel_size=0,   # dilation already applied above
            num_steps=steps,
            guidance_scale=guidance,
            seed=seed,
            paste=False,                 # paste-back happens below, at full resolution
            compensate=False,
            noise_offset=noise_offset,
            step_callback=step_callback,
        )

        out = []
        for result, original, m in zip(results, pil_images, pil_masks):
            result = result.resize((w0, h0), Image.Resampling.LANCZOS)
            if compensate:
                from .moebius_src.compensation_utils import paste_compensate
                result = paste_compensate(m, original, result, fac=1.1)
            elif paste:
                # Same blend as upstream _post_process, at input resolution.
                m_blur = np.asarray(m.convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))) / 255.0
                orig_np = np.asarray(original.convert("RGB")) / 255.0
                res_np = np.asarray(result) / 255.0
                blended = res_np * m_blur + (1 - m_blur) * orig_np
                result = Image.fromarray(np.uint8(blended * 255))
            out.append(result)

        return (pil_to_image_batch(out),)


NODE_CLASS_MAPPINGS = {
    "MoebiusModelLoader": MoebiusModelLoader,
    "MoebiusInpaint": MoebiusInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoebiusModelLoader": "Moebius Model Loader",
    "MoebiusInpaint": "Moebius Inpaint",
}
