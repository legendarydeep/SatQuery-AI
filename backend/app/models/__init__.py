"""
backend.app.models
==================
Real pretrained remote-sensing model interfaces for SatQuery AI:
1. GeoChat / GeoChat-KD (RS VLM Specialist)
2. Grounding DINO-Tiny (Text-Guided Zero-Shot Object Detector)
3. AdaptFormer-LEVIR-CD (Bi-Temporal Change Detection Specialist)
4. SAM 2 Tiny (Promptable Mask Segmentation Specialist)
"""

from app.models.base import BaseAIModel
from app.models.geochat import GeoChatModel
from app.models.grounding_dino import GroundingDINOModel
from app.models.adaptformer import AdaptFormerModel
from app.models.sam2 import SAM2Model

__all__ = [
    "BaseAIModel",
    "GeoChatModel",
    "GroundingDINOModel",
    "AdaptFormerModel",
    "SAM2Model",
]
