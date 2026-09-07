"""
satquery.change_detection.models
================================
Model abstraction layer supporting learned transformer architectures (ChangeFormer)
and deterministic classical spectral baselines.

Exports (public API)
--------------------
BaseChangeModel          – Abstract interface all models must implement.
ChangePrediction         – Standard prediction output contract.
ModelStatus              – Composite backward-compat status enum (REAL_MODEL / HEURISTIC_FALLBACK / ...).
ModelArtifactStatus      – Axis-1: checkpoint provenance.
RuntimeStatus            – Axis-2: execution health.
ValidationStatus         – Axis-3: domain validation.
DecisionTier             – Multi-tiered evidence confidence label.
ClassicalSpectralAdapter – Deterministic spectral baseline model.
ChangeFormerAdapter      – Learned transformer-based change detector.
get_change_model         – Factory function for model instantiation.
register_model           – Register a custom model class.
"""

from .base import (
    BaseChangeModel,
    ChangePrediction,
    ModelStatus,
    ModelArtifactStatus,
    RuntimeStatus,
    ValidationStatus,
    DecisionTier,
)
from .classical_adapter import ClassicalSpectralAdapter
from .changeformer import ChangeFormerAdapter
from .registry import get_change_model, register_model

__all__ = [
    "BaseChangeModel",
    "ChangePrediction",
    "ModelStatus",
    "ModelArtifactStatus",
    "RuntimeStatus",
    "ValidationStatus",
    "DecisionTier",
    "ClassicalSpectralAdapter",
    "ChangeFormerAdapter",
    "get_change_model",
    "register_model",
]
