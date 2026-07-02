# ComfyUI Moebius Inpainting

ComfyUI custom nodes for **[Moebius](https://github.com/hustvl/Moebius)** (hustvl, ECCV 2026) — a **0.22B-parameter diffusion inpainting specialist** that reaches ~10B-model (FLUX.1-Fill / SD3.5-class) quality at a fraction of the size: ~2–3 GB VRAM, ~20 denoising steps, no text encoder.

Moebius is **mask-driven object removal / inpainting**. There is **no text prompt** — conditioning comes from a fixed learned embedding table inside the model. You give it an image and a mask; it fills the masked region.

## What Moebius can and can't do

**Can:** remove anything you mask — objects, people, text/watermarks, blemishes, occlusions — and refill the hole with a plausible continuation of the surroundings, in well under a second. That single skill is the whole model, which is why 0.22B matches 10B-class generalists at it.

**Can't, by architecture (not by our wrapper):**
- **Prompt-guided inpainting** ("replace the car with a fountain") — Moebius has no text encoder at all; there is nothing to feed a prompt into.
- **Reference-guided inpainting** (insert the object/style from a second image) — the UNet has no image-conditioning input besides the masked source itself.

For those two jobs use a *generalist fill/edit model* (e.g. FLUX.1-Fill or a FLUX.2 Klein inpaint graph) — bigger and slower, but promptable. A good pattern is to combine them: Moebius for fast, seamless *removal*, the big model only when you need to *put something specific* in the hole.

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

### Which checkpoint?

All four are the **same architecture** trained/fine-tuned on different data — pick by content:

| Checkpoint | Fine-tuned on | Reach for it when |
|---|---|---|
| `pretrained` | large general corpus | **default** — everyday photos, arbitrary objects |
| `ft_places2` | Places2 (scenes, buildings, landscapes) | removing things from scenery, streets, interiors, backgrounds |
| `ft_celebahq` | CelebA-HQ (studio-style portraits) | portrait shots, head-and-shoulders framing |
| `ft_ffhq` | FFHQ (diverse face close-ups) | face close-ups — glasses, hands-in-front-of-face, occlusion repair |

If the face is small inside a bigger scene, `pretrained`/`ft_places2` usually beat the face models — the face checkpoints shine when the face fills the frame.

## GPU support (Blackwell / Ada / older)

The Moebius *student* model is **pure PyTorch** — no custom CUDA kernels, no flash-attention, no compiled wheels of its own. (The upstream repo's `flash-linear-attention` requirement belongs to the PixelHacker *teacher* used for distillation training; it is not part of inference and is not shipped here.)

GPU support is therefore exactly your torch build's support:

- **Blackwell (RTX 50xx, sm_120)** and **Ada (RTX 40xx, sm_89)**: works with the torch **≥ 2.7 + cu128** builds that current ComfyUI portable packages ship (verified: torch 2.9.1+cu128 arch list includes sm_120).
- Older NVIDIA cards, CPU, and Apple Silicon (`mps`): also work — pure PyTorch, fp32 by default.

## Example workflows

Drag a JSON from [`example_workflows/`](example_workflows/) onto the ComfyUI canvas. Both graphs carry note panels explaining the pipeline, the checkpoints and every parameter:

- [`moebius_inpaint_example.json`](example_workflows/moebius_inpaint_example.json) — **general object removal** (`pretrained`): `LoadImage` (paint the mask in the MaskEditor) → `Moebius Model Loader` → `Moebius Inpaint` → `SaveImage`.
- [`moebius_face_inpaint_example.json`](example_workflows/moebius_face_inpaint_example.json) — **face / portrait retouching** (`ft_ffhq`, mask_dilate 4): remove glasses, hands, occlusions from face close-ups.

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
- [x] 2026-07-02 — `download.py` (HF fetch, both checkpoints + shared VAE) + `conversions.py` (IMAGE/MASK ↔ PIL glue).
- [x] 2026-07-02 — Nodes (`MoebiusModelLoader`, `MoebiusInpaint`) + packaging; node defs verified under ComfyUI's module loader.
- [x] 2026-07-02 — Example workflow JSON; later expanded to two annotated examples (general removal + `ft_ffhq` face retouch) with in-graph note panels, plus the checkpoint guide and capabilities section in this README.
- [x] 2026-07-02 — Smoke tests green on RTX 5090 (Blackwell, torch 2.9.1+cu128): weights auto-download; checkpoint loads strict; 20-step 512×512 inpaint in **0.73 s** warm; same-seed rerun bit-identical; **mask polarity confirmed** (ComfyUI 1.0 = inpaint, no inversion); with `paste`, pixels >10 px from the mask are 100% bit-identical to the input; empty mask returns the input unchanged.
- [ ] In-ComfyUI graph test (loader → inpaint → save) — **pending a user test**: restart ComfyUI, drag in the example workflow, paint a mask, Queue.

## License

- This package: Apache-2.0.
- Vendored Moebius code (`moebius_src/`): Apache-2.0, © hustvl — see [NOTICE](NOTICE).
- Model weights: MIT (per the HuggingFace model cards).
