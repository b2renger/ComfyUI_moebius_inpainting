"""HuggingFace weight fetching for Moebius.

Deliberately ComfyUI-agnostic (no folder_paths import) so the standalone
smoke test can reuse it. The caller supplies the base directory
(ComfyUI/models/moebius in the nodes; any dir in tests).

Layouts (verified 2026-07-02):
  hustvl/Moebius      -> {pretrained,ft_places2,ft_celebahq,ft_ffhq}/diffusion_pytorch_model.bin
  hustvl/PixelHacker  -> vae/{config.json,diffusion_pytorch_model.bin}
"""
import os

MOEBIUS_REPO = "hustvl/Moebius"
VAE_REPO = "hustvl/PixelHacker"
CHECKPOINTS = ["pretrained", "ft_places2", "ft_celebahq", "ft_ffhq"]
CKPT_FILE = "diffusion_pytorch_model.bin"
VAE_FILES = ["config.json", "diffusion_pytorch_model.bin"]


def ensure_checkpoint(name: str, base_dir: str) -> str:
    """Return the local path of checkpoint `name`, downloading it if absent."""
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown Moebius checkpoint '{name}' (known: {CHECKPOINTS})")
    target = os.path.join(base_dir, name, CKPT_FILE)
    if not os.path.exists(target):
        from huggingface_hub import hf_hub_download
        print(f"[Moebius] downloading {MOEBIUS_REPO}/{name}/{CKPT_FILE} -> {base_dir} (~900 MB, first use only)")
        hf_hub_download(repo_id=MOEBIUS_REPO, filename=f"{name}/{CKPT_FILE}", local_dir=base_dir)
    return target


def ensure_vae(base_dir: str) -> str:
    """Return the local VAE directory (shared by all checkpoints), downloading if absent."""
    vae_dir = os.path.join(base_dir, "vae")
    for fn in VAE_FILES:
        if not os.path.exists(os.path.join(vae_dir, fn)):
            from huggingface_hub import hf_hub_download
            print(f"[Moebius] downloading {VAE_REPO}/vae/{fn} -> {base_dir}")
            hf_hub_download(repo_id=VAE_REPO, filename=f"vae/{fn}", local_dir=base_dir)
    return vae_dir
