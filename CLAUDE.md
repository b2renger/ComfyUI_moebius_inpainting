# ComfyUI_moebius_inpainting — project memory

ComfyUI custom-node package wrapping the **Moebius** 0.22B inpainting model (https://github.com/hustvl/Moebius). Canonical working copy: `D:\Code\ComfyUI_moebius_inpainting` (pushed to github.com/b2renger/ComfyUI_moebius_inpainting); the rig's `ComfyUI/custom_nodes/ComfyUI_moebius_inpainting` is a **plain clone of the GitHub repo** (production-test copy): dev + commit + push happen on D:, then `git pull` (or ComfyUI-Manager 'Update') on the J: clone. Don't edit the J: copy. Plan + dev log: [implementation_plan.md](implementation_plan.md); user-facing docs: [README.md](README.md). **Keep all three updated as development progresses.**

## Architecture decisions (the "why")

- **Wrapper, not native reimplementation** — the vendored subset of upstream code (`moebius_src/`) runs the original pipeline; ComfyUI nodes are thin glue. Rationale: the custom LλMI lambda-UNet is the model's whole value; reimplementing on comfy primitives is high-risk/no-reward.
- **★ No flash-linear-attention** — upstream `requirements.txt` lists `flash-linear-attention[cuda]` but it is imported ONLY by the PixelHacker *teacher* (`model_lib/nets/layers/gla/gla.py`, distillation training). The *student* UNet (`unet_lambda_prune_lite.py` + `layers/λ/vanillaλ.py`) is pure PyTorch (Conv/Linear/BatchNorm/einsum). So: no CUDA kernels, no wheels to build, portable to CPU/mps. Do NOT add fla as a dependency.
- **Blackwell/Ada wheels** — resolved to "nothing needed": pure PyTorch means GPU support == the host torch build. ComfyUI portable ships torch 2.9.1+cu128 whose arch list includes sm_120 (Blackwell) and covers Ada. **Never pin torch/torchvision in requirements.txt** — the upstream `torch==2.7.1+cu130` pin would clobber the user's ComfyUI torch.
- **V1 node API** (`NODE_CLASS_MAPPINGS`/`INPUT_TYPES`), not the V3 `io.ComfyNode` API — works on every ComfyUI version incl. older portable installs; it's what all wrapper repos (kijai etc.) use. Reference skills: https://github.com/jtydhr88/comfyui-custom-node-skills.
- **No text prompt by design** — conditioning is a fixed 20-entry `nn.Embedding` (ids 0–9 = conditional, 10–19 = unconditional for CFG), built once in the pipeline ctor. The inpaint node exposes no prompt field.
- **Weights: both paths** — loader dropdown lists local files under `ComfyUI/models/moebius/` (registered via `folder_paths.add_model_folder_path`) plus `(download) <name>` entries that `huggingface_hub.snapshot_download` on first use. UNets: `hustvl/Moebius` → `{pretrained,ft_places2,ft_celebahq,ft_ffhq}/diffusion_pytorch_model.bin`. VAE: `hustvl/PixelHacker` → `vae/{config.json,diffusion_pytorch_model.bin}` (shared by all checkpoints).

## Vendored code provenance (`moebius_src/`)

Copied from https://github.com/hustvl/Moebius @ `390735d867e6a7b337abad23af7f2e95eb8d5e63` (Apache-2.0, see NOTICE). Deliberate changes vs upstream — keep these when syncing:

- **Dropped**: teacher model (`unet_gla.py`, `layers/gla/`), EMA loading, training code, migan hooks, matplotlib latent visualization, `accelerate` import in `utils_infer.py` (was type-hint-only), `tqdm` (replaced by a step callback).
- **Renamed**: `layers/λ/vanillaλ.py` → `layers/lam/vanilla_lambda.py` (a literal `λ` path breaks some Windows/zip/packaging setups; imports patched).
- **Removed global side effects**: upstream set `torch._dynamo.config.suppress_errors = True` at import in two files — removed (it would silently change dynamo behavior for the whole ComfyUI process).
- **Silenced per-load debug prints** (`_init_continuous_input` in mix_transformer.py, block-type print in unet_lambda_prune_lite.py) — they fired ~18× on every model load.
- **`torch.meshgrid(..., indexing='ij')`** in layers/utils.py + vanilla_lambda.py — behavior-identical ('ij' is the legacy default), silences torch's deprecation warning.
- **`removal_model.py` rewritten**: upstream `build_removal_model` used `from model_lib import *` + `eval(model_type)`; ours imports the one student class explicitly and takes the config as a dict (no eval).
- **`pipeline.py` adapted** (`MoebiusPipeline`, single-batch): same math/order as upstream `RemovalSDXLPipeline_BatchMode` (mask binarize at 0.5 → resize-to-64-multiple LANCZOS → VAE encode ×2 → DDIM loop over 9-ch concat [noisy(4), mask(1), masked(4)] with CFG → decode → paste/compensate). Added: `step_callback(step, total)` (drives `comfy.utils.ProgressBar` + interrupt via raising in the callback), explicit `seed` param (upstream seeded globals with `retry`; its CLI `--seed` was dead code).

## Verified on rig (2026-07-02, RTX 5090 / torch 2.9.1+cu128 / diffusers 0.35.1)

- Standalone pipeline test PASS: checkpoint loads strict (`<All keys matched successfully>`, 226.0M params), 20-step 512×512 inpaint **0.73 s warm** (1.4 s cold incl. VAE), same-seed rerun **bit-identical**.
- **Mask polarity CONFIRMED empirically**: pipeline inpaints white(1.0)/keeps black — measured masked-region mean diff 23.4 vs kept-region 0.5 on the upstream sample. ComfyUI MASK (1.0 = edit) maps directly, **no inversion anywhere**.
- Node-level test PASS (headless, real `nodes.py` classes): loader dropdown, IMAGE/MASK conversion, node-side full-res paste (pixels >10 px from mask **100% bit-identical**, boundary blend ≤0.29 from the radius-3 Gaussian), empty-mask passthrough.
- **★ hf gotcha**: `hf_hub_download(local_dir=...)` creates `.cache/huggingface/` bookkeeping inside `models/moebius/` which leaked into `folder_paths.get_filename_list` — `_model_choices()` in nodes.py filters dot-dirs + non-checkpoint extensions. Keep that filter.

## Parity notes (for comparing against upstream CLI)

- Upstream CLI defaults: `--cfg 2.5`, `--pst true` (paste), `--cps false`, `--noise-offset 0.0357`, `--num-step 20`, fp32 pipeline (`dtype=torch.float`). Node defaults mirror these.
- `strength=0.99` hardcoded upstream (initial latent = pure noise only when `strength >= 1`; at 0.99 it's `add_noise` at the first timestep) — kept as the pipeline default, not exposed on the node.
- Seed parity: upstream seeds `random`/`np.random`/`torch.manual_seed` globally with `retry` (0 → seed 0). Our pipeline seeds the same way from the node's `seed` input, so identical seed/steps/cfg ⇒ identical output.
- VAE `scaling_factor` comes from `vae/config.json`; encode multiplies, decode divides — all inside vendored code, don't duplicate in node glue.

## Environment / test rig

- Rig: AN-5090-2, RTX 5090 (Blackwell), Windows. ComfyUI portable at `J:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\` — embedded python `python_embeded\python.exe` (3.12.10, torch 2.9.1+cu128, diffusers 0.35.1, einops/timm/cv2/omegaconf/huggingface_hub all present → `pip install -r requirements.txt` is a no-op here).
- Standalone smoke test: `python_embeded\python.exe custom_nodes\ComfyUI_moebius_inpainting\test_moebius.py` (auto-downloads weights on first run; writes PNGs into the repo's `_test_out/`, gitignored). Run it after any change to `moebius_src/` or the pipeline glue. Remember the J: copy tests what's PUSHED - push from D: and pull on J: first.
- diffusers compatibility: vendored code imports `diffusers.models.unets.unet_2d_condition`, `.unet_2d_blocks (get_down_block/get_mid_block/get_up_block)`, `transformer_2d`, `AdaGroupNorm`, etc. Verified against diffusers **0.35.1**; upstream targeted 0.38. If a future diffusers moves these, pin `<` the breaking version in requirements.txt.

## Example workflows (design decision, 2026-07-02)

Five graphs in `example_workflows/`. Moebius and FLUX-inpaint are **sibling tools for different jobs**, chosen per-task, NOT a forced pipeline — the owner flagged that shipping FLUX-only graphs in a Moebius repo implied a false pipeline. Resolution:
- **Moebius-only**: `moebius_inpaint_example` (general removal), `moebius_face_inpaint_example` (ft_ffhq, mask_dilate 4). No extra deps.
- **Combined (both nodes)**: `moebius_then_flux2_replace_example` — the ONE graph that legitimately uses both. Moebius erases the object to a clean plate; the cleaned IMAGE feeds FLUX's InpaintCrop **and** (via VAEEncode) the ReferenceLatent context. The SAME painted mask drives both stages. Rationale: FLUX ReferenceLatent replays the whole source image (old object included) as edit context, which `echoes` the object at the 4-step distilled setting; a Moebius clean-plate removes that echo. Genuinely better than FLUX-alone for REPLACE (not for ADD-to-empty-space).
- **Standalone FLUX alternatives** (no Moebius node): `flux2_klein_inpaint_prompt_example` (add by text), `flux2_klein_inpaint_reference_example` (add by reference). Note panels relabeled "STANDALONE - does NOT use the Moebius node".

Decision matrix: remove -> Moebius; add-to-empty (text/photo) -> standalone FLUX; replace-existing -> combined graph.

The three FLUX graphs need the **ComfyUI-Inpaint-CropAndStitch** pack (lquesada) + FLUX.2 Klein 9B models (flux-2-klein-9b-fp8 / qwen_3_8b_fp8mixed / flux2-vae) — all present on the AN-5090-2 rig (diffusion_models/, text_encoders/, vae/). FLUX graphs are hand-authored UI-format (litegraph) JSON mirroring ComfyQ's rig-proven `image_flux2_inpaint*` api.json; validated by a link-consistency check (every link defined once, consumed once, produced once, endpoints match) but NOT yet run end-to-end in the GUI - that's the owner's pending test.

## Conventions

- Heavy imports (`moebius_src`, diffusers) stay **out of module import time** — `nodes.py` imports them inside the node functions so ComfyUI startup isn't slowed and a broken dep doesn't kill node registration.
- Commit style: conventional prefixes (`docs:`, `vendor:`, `feat:`, `fix:`, `test:`), one logical unit per commit, keep the tree linear and human-readable.
- Commit at every logical unit; push to origin (github.com/b2renger/ComfyUI_moebius_inpainting) when a phase lands, per the owner's standing instruction.
- Docs discipline: every feature/fix updates README "Development status" + implementation_plan.md checkboxes in the same commit.
