# Implementation plan — Moebius → ComfyUI custom nodes

Approved plan (2026-07-02), updated with source-audit findings. Status checkboxes are kept current; the "why" behind decisions lives in [CLAUDE.md](CLAUDE.md).

## Context

[Moebius](https://github.com/hustvl/Moebius) (hustvl, ECCV; Apache-2.0 code, MIT weights) is a **0.22B-param diffusion inpainting specialist** that reaches ~10B-model (FLUX-Fill / SD3.5) quality at a fraction of the size/VRAM (~2–3 GB, 512×512). Upstream only ships a batch CLI (`python -m infer.infer_moebius`); there is no interactive/graph way to use it.

**Goal:** a standalone, publishable ComfyUI custom-node package exposing Moebius as **image + mask → inpainted image**. Weights obtainable both ways (local dropdown + HF download). Portable runtime (works beyond one CUDA/kernel combo). Must support at least **Blackwell (sm_120)** and **Ada (sm_89)**.

**Key trait:** no text prompt — conditioning is a fixed 20-entry learned `nn.Embedding` (half cond / half uncond for CFG). The inpaint node has no prompt field.

## Source-audit findings (2026-07-02, upstream @ `390735d8`)

These superseded parts of the original plan:

1. **`flash-linear-attention` is NOT needed.** It is imported only by the PixelHacker *teacher* (`model_lib/nets/layers/gla/gla.py`). The student UNet (`unet_lambda_prune_lite.py` + `layers/λ/vanillaλ.py`) is pure PyTorch. → The planned "pure-PyTorch fallback" is simply *the* implementation; no optional fast path, no wheels to build.
2. **Blackwell/Ada wheels: nothing to do** — pure PyTorch means GPU support == host torch build. Rig's ComfyUI portable ships torch 2.9.1+cu128 (arch list includes sm_120; Ada covered).
3. **Rig env already satisfies every dependency** (diffusers 0.35.1, einops, timm, cv2, pyyaml, huggingface_hub, omegaconf) — `requirements.txt` is belt-and-braces for other machines.
4. **Unicode dir confirmed** (`layers/λ/vanillaλ.py`) → renamed `layers/lam/vanilla_lambda.py` in the vendored copy.
5. **Global side effect found**: `torch._dynamo.config.suppress_errors = True` at import time (2 files) → stripped from the vendored copy.
6. **Student path needs**: `nets/{unet_lambda_prune_lite, unet_lambda_dwconv_blocks, utils}.py`, `layers/{utils, _efficientnet_blocks}.py`, `layers/lam/`, `layers/sana/{act,basic_modules,norms,utils}.py`, `layers/unet_blocks/{custom_down,custom_mid,custom_up,dw_resnet,mix_transformer}.py`, plus `removal/v1_2/{removal_model,pipeline,compensation_utils}.py`, `utils_infer.py`, `config/model_cfg/moebius.yaml`. Third-party: diffusers, einops, timm, cv2, numpy, PIL, yaml.
7. **HF layouts verified**: `hustvl/Moebius` → `{pretrained,ft_places2,ft_celebahq,ft_ffhq}/diffusion_pytorch_model.bin`; `hustvl/PixelHacker` → `vae/{config.json,diffusion_pytorch_model.bin}`.
8. **Upstream CLI defaults** (parity targets): cfg 2.5, paste on, compensate off, noise_offset 0.0357, 20 steps, fp32. Upstream's `--seed` flag is dead code; seeding happens via `retry` → global `torch.manual_seed`.

## Approach: wrapper (vendor Moebius student-inference code), not native reimplement

Vendor the minimal inference subset under `moebius_src/` (copy, not submodule — Manager doesn't init submodules), preserve Apache-2.0 attribution in NOTICE. Drop training code, teacher model, migan, matplotlib, accelerate, tqdm.

## Repo layout

```
__init__.py            # NODE_CLASS_MAPPINGS re-export + "moebius" model-folder registration; no heavy imports
nodes.py               # MoebiusModelLoader + MoebiusInpaint (V1 API; heavy imports inside functions)
conversions.py         # ComfyUI IMAGE/MASK <-> PIL glue
download.py            # huggingface_hub fetch into models/moebius/
moebius_src/           # vendored inference subset (see CLAUDE.md provenance)
example_workflows/moebius_inpaint_example.json
test_moebius.py        # standalone smoke test (no ComfyUI needed)
requirements.txt       # curated; NEVER pins torch
pyproject.toml         # [project] + [tool.comfy] registry metadata
README.md / CLAUDE.md / implementation_plan.md / LICENSE / NOTICE
```

## Node set

1. **`MoebiusModelLoader`** → `MOEBIUS_PIPE` (pipeline object).
   - `model_name`: local files under `models/moebius/` + `(download) pretrained|ft_places2|ft_celebahq|ft_ffhq`.
   - `dtype`: fp32 (default; upstream-recommended) / fp16 / bf16.
   - Builds: student UNet from vendored `moebius.yaml` + `torch.load` state dict; `AutoencoderKL.from_pretrained(models/moebius/vae)`; `DDIMScheduler(beta_start=0.00085, beta_end=0.012, scaled_linear, 1000, clip_sample=False)`. Device from `comfy.model_management.get_torch_device()`.
2. **`MoebiusInpaint`** → `IMAGE`.
   - Inputs: `pipe`, `image (IMAGE)`, `mask (MASK)`, `steps (20)`, `guidance (2.5)`, `seed`, `image_size (512, step 64)`, `mask_dilate (0)`, `paste (true)`, `compensate (false)`, `noise_offset (0.0357)`.
   - Per batch item: IMAGE/MASK → PIL → vendored pipeline (mask binarize ≥0.5, resize to 64-multiple, VAE encode ×2, DDIM loop on 9-ch concat with CFG, decode, paste-back) → IMAGE. ProgressBar + interrupt via the pipeline's step callback.
   - Mask polarity: ComfyUI MASK 1.0 = edit region; Moebius white = inpaint → same convention, no inversion (verify on rig).

## Dependency strategy

`requirements.txt`: `diffusers`, `einops`, `timm`, `opencv-python`, `pyyaml`, `huggingface_hub`, `numpy`, `Pillow` (most ship with ComfyUI already). **No torch/torchvision pin** — upstream's `torch==2.7.1+cu130` pin must never reach users' installs. No `flash-linear-attention`, no `accelerate`, no training deps.

## Status

- [x] Phase 0 — source audit (findings above)
- [x] Phase 1 — repo scaffold + docs (this commit)
- [ ] Phase 2 — vendor `moebius_src/` (import fixes, λ→lam rename, dynamo-side-effect strip, pipeline adaptation w/ step callback + seed param)
- [ ] Phase 3 — `download.py` + `conversions.py`
- [ ] Phase 4 — `nodes.py` + `__init__.py` + packaging (`requirements.txt`, `pyproject.toml`)
- [ ] Phase 5 — example workflow JSON + `test_moebius.py`
- [ ] Phase 6 — smoke test on rig (RTX 5090)
- [ ] Phase 7 — in-ComfyUI graph test (owner) → publish

## Verification

1. **Standalone smoke test** (rig, embedded python): `python_embeded\python.exe custom_nodes\ComfyUI_moebius_inpainting\test_moebius.py` — downloads weights if absent, inpaints a sample image+mask from the upstream repo's `data/`, writes `_test_out/*.png`. Run with a fixed seed twice → identical outputs.
2. **Parity vs upstream** (optional, needs upstream clone + its deps): same image/mask/seed/steps/cfg through `python -m infer.infer_moebius` and through our pipeline → outputs should match.
3. **In ComfyUI**: restart, build `LoadImage(+mask) → MoebiusModelLoader → MoebiusInpaint → SaveImage` (or drag the example workflow). Confirm: masked region inpainted (polarity!), ProgressBar advances, Cancel interrupts, `(download)` entry fetches on first use, startup not slowed.
