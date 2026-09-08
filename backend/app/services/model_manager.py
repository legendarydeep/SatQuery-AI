"""
backend.app.services.model_manager
==================================
Central Model Registry and Execution Manager for SatQuery AI.
Orchestrates the 4 real pretrained models:
1. GeoChat / GeoChat-KD
2. Grounding DINO-Tiny
3. AdaptFormer-LEVIR-CD
4. SAM 2 Tiny
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.models.adaptformer import AdaptFormerModel
from app.models.geochat import GeoChatModel
from app.models.grounding_dino import GroundingDINOModel
from app.models.sam2 import SAM2Model

logger = logging.getLogger(__name__)

TASK_REGISTRY: Dict[str, str] = {
    "scene_understanding": "geochat",
    "vqa": "geochat",
    "captioning": "geochat",
    "grounding": "grounding_dino",
    "object_detection": "grounding_dino",
    "segmentation": "sam2",
    "mask_refinement": "sam2",
    "change_detection": "adaptformer",
    "bi_temporal_change": "adaptformer",
    "spectral_analysis": "spectral_engine",
    "sar_analysis": "sar_engine",
}


class ModelManager:
    """
    Manages loading, execution, status, and caching of the 4 specialist AI models.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._geochat: Optional[GeoChatModel] = None
        self._grounding_dino: Optional[GroundingDINOModel] = None
        self._adaptformer: Optional[AdaptFormerModel] = None
        self._sam2: Optional[SAM2Model] = None

    @property
    def geochat(self) -> GeoChatModel:
        if self._geochat is None:
            self._geochat = GeoChatModel(device=self.device)
            self._geochat.load()
        return self._geochat

    @property
    def grounding_dino(self) -> GroundingDINOModel:
        if self._grounding_dino is None:
            self._grounding_dino = GroundingDINOModel(device=self.device)
            self._grounding_dino.load()
        return self._grounding_dino

    @property
    def adaptformer(self) -> AdaptFormerModel:
        if self._adaptformer is None:
            self._adaptformer = AdaptFormerModel(device=self.device)
            self._adaptformer.load()
        return self._adaptformer

    @property
    def sam2(self) -> SAM2Model:
        if self._sam2 is None:
            self._sam2 = SAM2Model(device=self.device)
            self._sam2.load()
        return self._sam2

    def get_model(self, model_key: str) -> Any:
        key = model_key.lower().strip()
        if "geochat" in key:
            return self.geochat
        elif "dino" in key or "grounding" in key:
            return self.grounding_dino
        elif "adaptformer" in key or "change" in key:
            return self.adaptformer
        elif "sam" in key or "segment" in key:
            return self.sam2
        raise ValueError(f"Unknown model identifier: {model_key}")

    def list_models_status(self) -> list[Dict[str, Any]]:
        return [
            {
                "id": "geochat-7b",
                "name": "GeoChat-7B",
                "role": "Remote Sensing Vision-Language Copilot & Scene Understanding",
                "params": "7.1B",
                "architecture": "LLaVA-1.5 RS Adapted (ViT-G/14 + Vicuna-7B)",
                "status": "ready" if self._geochat and self._geochat.is_loaded else "standby",
                "hf_repo": "MBZUAI/geochat-7b",
                "real_model": True,
            },
            {
                "id": "grounding-dino-tiny",
                "name": "Grounding DINO-Tiny",
                "role": "Text-Guided Zero-Shot Object Detector",
                "params": "172M",
                "architecture": "Swin-T + Deformable Transformer Cross-Attention",
                "status": "ready" if self._grounding_dino and self._grounding_dino.is_loaded else "standby",
                "hf_repo": "IDEA-Research/grounding-dino-tiny",
                "real_model": True,
            },
            {
                "id": "adaptformer-levir-cd",
                "name": "AdaptFormer-LEVIR-CD",
                "role": "Bi-Temporal Remote Sensing Change Detection",
                "params": "12.5M",
                "architecture": "ViT-Adapter Bi-Temporal Difference Network",
                "status": "ready" if self._adaptformer and self._adaptformer.is_loaded else "standby",
                "hf_repo": "deepang/adaptformer-LEVIR-CD",
                "real_model": True,
            },
            {
                "id": "sam-2-tiny",
                "name": "SAM 2 Tiny",
                "role": "Promptable Pixel-Accurate Mask Segmenter",
                "params": "38.9M",
                "architecture": "Hiera-Tiny Hierarchical Vision Transformer",
                "status": "ready" if self._sam2 and self._sam2.is_loaded else "standby",
                "hf_repo": "facebook/sam2-hiera-tiny",
                "real_model": True,
            },
        ]


model_manager = ModelManager()
