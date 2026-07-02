# ComfyUI Moebius Inpainting

ComfyUI custom nodes for **[Moebius](https://github.com/hustvl/Moebius)** (hustvl, ECCV 2026) — a **0.22B-parameter diffusion inpainting specialist** that reaches ~10B-model (FLUX.1-Fill / SD3.5-class) quality at a fraction of the size: ~2–3 GB VRAM, ~20 denoising steps, no text encoder.

Moebius is **mask-driven object removal / inpainting**. There is **no text prompt** — conditioning comes from a fixed learned embedding table inside the model. You give it an image and a mask; it fills the masked region.

## How it works (and why there is no prompt)

Moebius is a latent diffusion model, and it **is generative** — it invents brand-new pixels in the hole (change the seed and the fill changes). What it lacks is a *language* channel to tell it **what** to invent; instead, the surrounding image is the instruction.

At every denoising step the UNet sees three things stacked together (9 latent channels): the current **noisy latent** of the whole image, your **mask**, and the latent of the image **with the hole blanked out** (the context). Starting from pure noise inside the mask, ~20 denoising steps pull that region toward "whatever is statistically plausible given the surrounding pixels" — continuing textures, edges, lighting, geometry, even face structure. It learned that prior from millions of (image, random mask) pairs, distilled from the 10B-parameter PixelHacker teacher: the 0.22B student is trained to match the teacher's denoising predictions, which is how it keeps 10B-class quality at one narrow job.

Where a promptable model (FLUX-class) feeds a text encoder's output into cross-attention, Moebius replaces that entire subsystem with **20 fixed embedding vectors learned during training** (10 "do the task" + 10 "unconditional" for classifier-free guidance — the `guidance` knob interpolates between them). Think of it as a task instruction permanently baked into the weights, always meaning *"continue this scene naturally"*. There is simply no input where a prompt or a reference image could enter — which is precisely the trade that makes it 50× smaller and >15× faster than a generalist.

## What Moebius can and can't do

**Can:** remove anything you mask — objects, people, text/watermarks, blemishes, occlusions — and refill the hole with a plausible continuation of the surroundings, in well under a second. That single skill is the whole model, which is why 0.22B matches 10B-class generalists at it.

**Can't, by architecture (not by our wrapper):**
- **Prompt-guided inpainting** ("replace the car with a fountain") — Moebius has no text encoder at all; there is nothing to feed a prompt into.
- **Reference-guided inpainting** (insert the object/style from a second image) — the UNet has no image-conditioning input besides the masked source itself.

For those two jobs use a *generalist fill/edit model* — bigger and slower, but promptable. This repo ships ready-made graphs built on **FLUX.2 Klein 9B** so the folder covers the full spectrum. Note the FLUX graphs are **their own pipelines** — most are *alternatives* to Moebius, run one or the other; the exception is the "replace" graph, which genuinely uses **both** nodes (Moebius cleans, FLUX fills):

| You want to… | Use | Moebius node? |
|---|---|---|
| **remove** something (fill with background) | **Moebius** — fastest, ~1 GB | ✅ only Moebius |
| **add** something to empty/background space, *described* | `flux2_klein_inpaint_prompt_example` | ❌ FLUX-only alternative |
| **add** something from *another photo* | `flux2_klein_inpaint_reference_example` | ❌ FLUX-only alternative |
| **replace** an existing object with a *described* one | `moebius_then_flux2_replace_example` | ✅ **both** — Moebius removes → FLUX fills |

Why the last row chains them: FLUX's `ReferenceLatent` feeds the whole source image (the object you want gone included) back as edit context, so at low step counts it can "echo" the old object. Removing it with Moebius first gives FLUX a clean plate to generate on. For plain *add-to-empty-space* that pre-pass is unnecessary — use the standalone FLUX graphs.

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

Drag a JSON from [`example_workflows/`](example_workflows/) onto the ComfyUI canvas. Every graph carries on-canvas note panels explaining the pipeline, the checkpoints and every parameter.

**Moebius-only** (removal — no extra dependencies):

- [`moebius_inpaint_example.json`](example_workflows/moebius_inpaint_example.json) — **general object removal** (`pretrained`): `LoadImage` (paint the mask in the MaskEditor) → `Moebius Model Loader` → `Moebius Inpaint` → `SaveImage`.
- [`moebius_face_inpaint_example.json`](example_workflows/moebius_face_inpaint_example.json) — **face / portrait retouching** (`ft_ffhq`, mask_dilate 4): remove glasses, hands, occlusions from face close-ups.

A **combined graph** uses both nodes — the only one where FLUX and Moebius work together:

- [`moebius_then_flux2_replace_example.json`](example_workflows/moebius_then_flux2_replace_example.json) — **replace an existing object**: paint it once → `Moebius Inpaint` erases it to a clean plate → FLUX.2 fills the region from your prompt (fed the *cleaned* image as context, so no ghosting of the old object).

Two **standalone FLUX.2 graphs** are *alternatives* to Moebius (they contain **no Moebius node** — use them instead of Moebius, not with it):

- [`flux2_klein_inpaint_prompt_example.json`](example_workflows/flux2_klein_inpaint_prompt_example.json) — **add with a text prompt**: paint a region, describe what should appear there.
- [`flux2_klein_inpaint_reference_example.json`](example_workflows/flux2_klein_inpaint_reference_example.json) — **add from a reference image**: paint *where*, a second image supplies *what*.

> **Dependencies for the FLUX.2 graphs only** (the two Moebius-only graphs need none of this): all three FLUX graphs use the `InpaintCropImproved` / `InpaintStitchImproved` nodes from the **[ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)** pack by lquesada — install via **ComfyUI-Manager** (search "Inpaint-CropAndStitch") or clone into `custom_nodes/`:
> ```bash
> cd ComfyUI/custom_nodes
> git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git
> ```
> They also need the FLUX.2 Klein 9B models (diffusion model, Qwen3 text encoder, VAE) — exact files and download locations are in each graph's on-canvas note panel.

## Parameter guide

- **steps** (default 20) — DDIM denoising steps.
- **guidance** (default 2.5) — classifier-free guidance; upstream uses 2.0–2.5. Higher = stronger removal, but can hallucinate.
- **image_size** (default 512) — the **square** side (a multiple of 64) the model runs at. Moebius is trained at 512×512 and its attention only handles square latents, so the node resizes your image to `image_size × image_size` for processing, then resizes the result back to the original width×height. **Non-square images are supported** — they're squished to square internally and un-squished on output (this is exactly what upstream does), so the aspect ratio you put in is the aspect ratio you get back.
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
- [x] 2026-07-02 — Example workflows (5 total): 2 Moebius-only (general removal + `ft_ffhq` face retouch), 1 combined `moebius_then_flux2_replace` (Moebius clean-plate → FLUX.2 fill), 2 standalone FLUX.2 alternatives (prompt / reference). All carry in-graph note panels; all pass link-consistency validation. README gained the how-it-works, checkpoint guide, capabilities, and remove/add/replace decision table.
- [x] 2026-07-02 — Smoke tests green on RTX 5090 (Blackwell, torch 2.9.1+cu128): weights auto-download; checkpoint loads strict; 20-step 512×512 inpaint in **0.73 s** warm; same-seed rerun bit-identical; **mask polarity confirmed** (ComfyUI 1.0 = inpaint, no inversion); with `paste`, pixels >10 px from the mask are 100% bit-identical to the input; empty mask returns the input unchanged.
- [x] 2026-07-02 — **In-ComfyUI test surfaced a real bug** (non-square images → `EinopsError`, the lambda attention is square-only). Fixed: the pipeline now processes at a square `image_size` (upstream's own behavior) and the node resizes back to the original aspect. Verified on the rig across 512×512 / 640×512 / 512×640 / 768×512 / 896×512 / 500×700 and a real 512×384 crop (dims preserved, far-from-mask pixels 100% identical, deterministic); non-square regression added to `test_moebius.py`.
- [ ] In-ComfyUI graph test of the **combined + standalone FLUX graphs** — **pending a user test** (need the CropAndStitch pack + FLUX.2 models, all present on the AN-5090-2 rig).

## License

- This package: Apache-2.0.
- Vendored Moebius code (`moebius_src/`): Apache-2.0, © hustvl — see [NOTICE](NOTICE).
- Model weights: MIT (per the HuggingFace model cards).
