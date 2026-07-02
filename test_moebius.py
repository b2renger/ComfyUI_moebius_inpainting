"""Standalone smoke test for the vendored Moebius pipeline (no ComfyUI needed).

Run with any python that has the requirements installed, e.g. on a ComfyUI
portable install:

    python_embeded\\python.exe custom_nodes\\ComfyUI_moebius_inpainting\\test_moebius.py

What it does:
  1. ensures weights exist (downloads ~900 MB checkpoint + VAE on first run)
     into ComfyUI/models/moebius when run in-place, else ./_weights
  2. builds the pipeline and inpaints a test image+mask (upstream repo samples,
     fetched on demand; or pass --image/--mask)
  3. checks: masked region changed, unmasked corner untouched (paste),
     same-seed rerun is bit-identical
  4. writes input/mask/outputs into _test_out/ next to this file

Exit code 0 = all checks passed.
"""
import argparse
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from download import ensure_checkpoint, ensure_vae, CHECKPOINTS  # noqa: E402

SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/hustvl/Moebius/main/data/images/0.png"
SAMPLE_MASK_URL = "https://raw.githubusercontent.com/hustvl/Moebius/main/data/masks/000000.png"


def default_models_dir():
    # custom_nodes/ComfyUI_moebius_inpainting -> ComfyUI/models/moebius
    comfy_models = os.path.abspath(os.path.join(HERE, "..", "..", "models"))
    if os.path.isdir(comfy_models):
        return os.path.join(comfy_models, "moebius")
    return os.path.join(HERE, "_weights")


def fetch(url, path):
    if not os.path.exists(path):
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="pretrained", choices=CHECKPOINTS)
    ap.add_argument("--models-dir", default=default_models_dir())
    ap.add_argument("--image", default=None, help="test image (default: upstream sample)")
    ap.add_argument("--mask", default=None, help="test mask, white = inpaint (default: upstream sample)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")

    out_dir = os.path.join(HERE, "_test_out")
    os.makedirs(out_dir, exist_ok=True)

    # --- weights ---
    ckpt = ensure_checkpoint(args.checkpoint, args.models_dir)
    vae_dir = ensure_vae(args.models_dir)
    print(f"checkpoint: {ckpt}")

    # --- test data ---
    image_path = args.image or fetch(SAMPLE_IMAGE_URL, os.path.join(out_dir, "sample_image.png"))
    mask_path = args.mask or fetch(SAMPLE_MASK_URL, os.path.join(out_dir, "sample_mask.png"))
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    print(f"image: {image.size}  mask: {mask.size}")

    # --- pipeline ---
    from moebius_src import (
        MOEBIUS_CONFIG_PATH, MoebiusPipeline, build_removal_model, load_removal_model)
    from diffusers import AutoencoderKL, DDIMScheduler

    t0 = time.time()
    model = build_removal_model(MOEBIUS_CONFIG_PATH, num_embeddings=20)
    print(load_removal_model(model, ckpt, device="cpu"))
    vae = AutoencoderKL.from_pretrained(vae_dir, use_safetensors=False)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False)
    pipe = MoebiusPipeline(model, vae, scheduler, device=device, dtype=torch.float32)
    print(f"pipeline built in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    def run():
        steps_seen = []
        t = time.time()
        result = pipe(
            [image], [mask],
            image_size=512, num_steps=args.steps, guidance_scale=args.cfg,
            seed=args.seed, paste=True, noise_offset=0.0357,
            step_callback=lambda i, n: steps_seen.append(i))
        dt = time.time() - t
        assert len(steps_seen) >= args.steps, f"step_callback fired {len(steps_seen)}x"
        return result[0], dt

    result1, dt1 = run()
    result2, dt2 = run()
    print(f"inference: {dt1:.2f}s / rerun {dt2:.2f}s ({args.steps} steps)")

    image.save(os.path.join(out_dir, "input.png"))
    mask.save(os.path.join(out_dir, "mask.png"))
    result1.save(os.path.join(out_dir, "output.png"))

    # --- checks ---
    a1, a2 = np.asarray(result1), np.asarray(result2)
    ok = True

    if a1.shape != a2.shape or not (a1 == a2).all():
        print("FAIL: same-seed rerun is not deterministic")
        ok = False
    else:
        print("ok: same-seed rerun bit-identical")

    # the pipeline processes at a SQUARE resolution (image_size x image_size),
    # so compare against the input squared the same way
    ref_img = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    ref_mask = mask.resize((512, 512), Image.Resampling.NEAREST)
    ref = np.asarray(ref_img).astype(np.int16)
    m = np.asarray(ref_mask.point(lambda x: 0 if x < 128 else 255).convert("L")) >= 128
    if m.sum() == 0:
        print("FAIL: sample mask is empty")
        ok = False
    else:
        diff = np.abs(a1.astype(np.int16) - ref).mean(axis=-1)
        changed = (diff[m] > 8).mean()
        print(f"masked-region pixels changed (>8/255): {changed:.1%}")
        if changed < 0.2:
            print("FAIL: masked region barely changed - inpainting suspect")
            ok = False
        # far-from-mask pixels must be untouched (paste blend, blur radius 3)
        grown = ref_mask.point(lambda x: 0 if x < 128 else 255).filter(ImageFilter.MaxFilter(31))
        far = ~(np.asarray(grown) >= 128)
        untouched = (diff[far] == 0).mean() if far.sum() else 1.0
        print(f"unmasked (far) pixels identical: {untouched:.1%}")
        if untouched < 0.99:
            print("FAIL: paste-back should keep unmasked pixels identical")
            ok = False

    # --- non-square regression (the lambda attention is square-only; the
    #     pipeline must square the input internally or it raises EinopsError) ---
    ns_img = image.convert("RGB").resize((640, 512), Image.Resampling.LANCZOS)  # 5:4 landscape
    ns_mask = mask.resize((640, 512), Image.Resampling.NEAREST)
    try:
        ns_out = pipe([ns_img], [ns_mask], image_size=512, num_steps=4,
                      guidance_scale=args.cfg, seed=args.seed, paste=True)[0]
        print(f"ok: non-square 640x512 input ran -> {ns_out.size}")
    except Exception as e:
        print(f"FAIL: non-square input raised {type(e).__name__}: {str(e).splitlines()[0]}")
        ok = False

    print("PASS" if ok else "FAIL", f"- outputs in {out_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
