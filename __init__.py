"""ComfyUI Moebius Inpainting — entry point.

Registers the models/moebius folder and the node mappings. Keep this file
free of heavy imports (torch is already resident in ComfyUI; moebius_src /
diffusers load lazily inside the node functions).
"""
import os

import folder_paths

_moebius_dir = os.path.join(folder_paths.models_dir, "moebius")
os.makedirs(_moebius_dir, exist_ok=True)
folder_paths.add_model_folder_path("moebius", _moebius_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
