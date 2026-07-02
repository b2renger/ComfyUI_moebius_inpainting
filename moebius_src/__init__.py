# Vendored subset of https://github.com/hustvl/Moebius (student-model inference
# only). Provenance + list of modifications: see NOTICE and CLAUDE.md.
import os

from .removal_model import (
    RemovalModel,
    build_removal_model,
    load_cfg,
    load_removal_model,
)
from .pipeline import MoebiusPipeline

# Bundled upstream model config for the published checkpoints.
MOEBIUS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moebius.yaml")

__all__ = [
    "RemovalModel",
    "build_removal_model",
    "load_cfg",
    "load_removal_model",
    "MoebiusPipeline",
    "MOEBIUS_CONFIG_PATH",
]
