# ComfyUI Moebius Inpainting

ComfyUI custom nodes for **[Moebius](https://github.com/hustvl/Moebius)** (hustvl, ECCV 2026) — a **0.22B-parameter diffusion inpainting specialist** that reaches ~10B-model (FLUX.1-Fill / SD3.5-class) quality at a fraction of the size: ~2–3 GB VRAM, ~20 denoising steps, no text encoder.

Moebius is **mask-driven object removal / inpainting**. There is **no text prompt** — conditioning comes from a fixed learned embedding table inside the model. You give it an image and a mask; it fills the masked region.

## Nodes

| Node | What it does |
|---|---|
| **Moebius Model Loader** | Loads a Moebius checkpoint + the PixelHacker VAE. The dropdown lists local checkpoints in `ComfyUI/models/moebius/` **and** `(download)` entries that fetch the official weights from HuggingFace on first use. |
| **Moebius Inpaint** | `IMAGE` + `MASK` → inpainted `IMAGE`. Knobs: steps, guidance (CFG), seed, resolution, mask dilation, paste-back blending. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/b2renger/ComfyUI_moebius_inpainting
# portable install:
..\..\python_embeded\python.exe -m pip install -r ComfyUI_moebius_inpainting\requirements.txt
# (regular install: pip install -r ComfyUI_moebius_inpainting/requirements.txt)
```

Restart ComfyUI. On most installs no new packages are needed (see requirements — everything is a common dependency; **torch is intentionally not pinned**).

## Models

Either pick a `(download)` entry in the loader (fetched automatically into `ComfyUI/models/moebius/`), or place files manually:

```
ComfyUI/models/moebius/
├── pretrained/diffusion_pytorch_model.bin      # general inpainting (from hustvl/Moebius)
├── ft_places2/diffusion_pytorch_model.bin      # natural scenes
├── ft_celebahq/diffusion_pytorch_model.bin     # portraits
├── ft_ffhq/diffusion_pytorch_model.bin         # faces
└── vae/                                        # from hustvl/PixelHacker
    ├── config.json
    └── diffusion_pytorch_model.bin
```

Sources: [hustvl/Moebius](https://huggingface.co/hustvl/Moebius) (UNet checkpoints, ~900 MB each, MIT) and [hustvl/PixelHacker](https://huggingface.co/hustvl/PixelHacker) (`vae/`, ~335 MB). The VAE is shared by all checkpoints and auto-downloaded alongside any `(download)` selection.

## GPU support (Blackwell / Ada / older)

The Moebius *student* model is **pure PyTorch** — no custom CUDA kernels, no flash-attention, no compiled wheels of its own. (The upstream repo's `flash-linear-attention` requirement belongs to the PixelHacker *teacher* used for distillation training; it is not part of inference and is not shipped here.)

GPU support is therefore exactly your torch build's support:

- **Blackwell (RTX 50xx, sm_120)** and **Ada (RTX 40xx, sm_89)**: works with the torch **≥ 2.7 + cu128** builds that current ComfyUI portable packages ship (verified: torch 2.9.1+cu128 arch list includes sm_120).
- Older NVIDIA cards, CPU, and Apple Silicon (`mps`): also work — pure PyTorch, fp32 by default.

## Example workflow

Drag [`example_workflows/moebius_inpaint_example.json`](example_workflows/moebius_inpaint_example.json) onto the ComfyUI canvas: `LoadImage` (draw the mask in the mask editor) → `Moebius Model Loader` → `Moebius Inpaint` → `SaveImage`.

## Parameter guide

- **steps** (default 20) — DDIM denoising steps.
- **guidance** (default 2.5) — classifier-free guidance; upstream uses 2.0–2.5. Higher = stronger removal, but can hallucinate.
- **image_size** (default 512) — processing resolution (short side; snapped to a multiple of 64). The model is trained at 512; the output is resized back and (with `paste`) blended into the original at full resolution.
- **mask_dilate** (default 0) — grow the mask by N pixels before inpainting; useful when the mask hugs the object too tightly.
- **paste** (default on) — Gaussian-blend the inpainted region back into the *original-resolution* image, so unmasked pixels stay pixel-identical.
- **seed** — reproducibility.
- **fp16/bf16** — available in the loader; upstream recommends fp32 for best quality (the model is small enough that fp32 is cheap).

## Development status

Progress log (kept up to date — see [implementation_plan.md](implementation_plan.md) for the full plan):

- [x] 2026-07-02 — Upstream source audited: student UNet confirmed pure PyTorch (no `fla` dependency); inference path mapped (`RemovalSDXLPipeline_BatchMode`); HF weight layouts verified.
- [x] 2026-07-02 — Repo scaffold + docs.
- [x] 2026-07-02 — Vendored minimal inference subset under `moebius_src/` (Apache-2.0, see NOTICE); imports + model construction verified against diffusers 0.35.1 (226.0M params).
- [ ] `download.py` (HF fetch) + `conversions.py` (IMAGE/MASK ↔ PIL glue).
- [ ] Nodes (`MoebiusModelLoader`, `MoebiusInpaint`) + packaging.
- [ ] Example workflow JSON.
- [ ] Standalone smoke test on the rig (RTX 5090, torch 2.9.1+cu128).
- [ ] In-ComfyUI graph test (loader → inpaint → save).

## License

- This package: Apache-2.0.
- Vendored Moebius code (`moebius_src/`): Apache-2.0, © hustvl — see [NOTICE](NOTICE).
- Model weights: MIT (per the HuggingFace model cards).
