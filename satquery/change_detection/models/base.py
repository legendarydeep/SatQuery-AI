"""
satquery.change_detection.models.base
=====================================
Abstract base interfaces, prediction contracts, and tri-axis model provenance
tracking for SatQuery AI change detection engines.

Three orthogonal status axes
----------------------------
  Artifact   — Is the checkpoint authentic? (ModelArtifactStatus)
  Runtime    — Can we execute it?           (RuntimeStatus)
  Validation — Does it perform acceptably?  (ValidationStatus)

A composite ModelStatus enum is provided for backward-compatibility with the
pipeline, Tool 3, and tests that expect a single status label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from satquery.core.raster_io import RasterData


# ---------------------------------------------------------------------------
# Axis 1 — Artifact provenance
# ---------------------------------------------------------------------------

class ModelArtifactStatus(str, Enum):
    """
    Provenance and authenticity of the model weight artifact.

    REAL_CHECKPOINT            – Authentic official pretrained weights.
    DOMAIN_ADAPTED_CHECKPOINT  – Fine-tuned / calibrated checkpoint.
    TEST_FIXTURE               – Synthetic weights for CI tests only (NEVER production).
    MOCK                       – Stub placeholder (strictly non-production).
    UNAVAILABLE                – Checkpoint file is missing or unreadable.
    """
    REAL_CHECKPOINT = "real_checkpoint"
    DOMAIN_ADAPTED_CHECKPOINT = "domain_adapted_checkpoint"
    DETERMINISTIC_ALGORITHM = "deterministic_algorithm"
    TEST_FIXTURE = "test_fixture"
    MOCK = "mock"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Axis 2 — Runtime execution
# ---------------------------------------------------------------------------

class RuntimeStatus(str, Enum):
    """
    Execution readiness and health of the model runtime engine.

    LOADED            – Weights and compute graph loaded into memory/device.
    INFERENCE_SUCCESS – Forward pass completed without numerical faults.
    INFERENCE_FAILED  – Execution error or non-finite tensor output.
    INCOMPATIBLE      – Missing dependency (PyTorch / CUDA / ONNX).
    """
    LOADED = "loaded"
    INFERENCE_SUCCESS = "inference_success"
    INFERENCE_FAILED = "inference_failed"
    INCOMPATIBLE = "incompatible"


# ---------------------------------------------------------------------------
# Axis 3 — Domain validation
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    """
    Empirical validation on held-out target-domain data.

    VALIDATED_ON_TARGET_DOMAIN – Meets baseline thresholds on labeled split.
    PROVISIONAL_VALIDATION     – Basic sanity checks and non-zero response verified.
    UNVALIDATED                – Raw weights loaded without empirical domain validation.
    VALIDATION_FAILED          – Failed to achieve minimum performance thresholds.
    """
    VALIDATED_ON_TARGET_DOMAIN = "validated_on_target_domain"
    PROVISIONAL_VALIDATION = "provisional_validation"
    UNVALIDATED = "unvalidated"
    VALIDATION_FAILED = "validation_failed"


# ---------------------------------------------------------------------------
# Composite backward-compatibility enum
# ---------------------------------------------------------------------------

class ModelStatus(str, Enum):
    """
    Single composite model status enum for backward-compatibility.

    Derived from the tri-axis (Artifact × Runtime × Validation) system.
    Exposed at package level so the pipeline, Tool 3, and tests do not need
    to reason about all three axes simultaneously.

    REAL_MODEL          – Authentic weights + successful inference.
    HEURISTIC_FALLBACK  – Learned model unavailable; classical baseline used.
    CLASSICAL_ALGORITHM – Deterministic classical pipeline (no learned model).
    UNAVAILABLE         – No inference path available (hard failure).
    """
    REAL_MODEL = "real_model"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    CLASSICAL_ALGORITHM = "classical_algorithm"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Decision tier
# ---------------------------------------------------------------------------

class DecisionTier(str, Enum):
    """
    Multi-tiered evidence confidence decision.

    VERIFIED    – High model confidence + corroborating spectral evidence.
    PROBABLE    – High model confidence but ambiguous spectral context.
    UNCERTAIN   – Contradiction between learned features and physical indicators.
    REJECTED    – Insignificant / transient noise below threshold.
    UNAVAILABLE – Model failure or unreadable data.
    """
    VERIFIED = "VERIFIED"
    PROBABLE = "PROBABLE"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Prediction contract
# ---------------------------------------------------------------------------

@dataclass
class ChangePrediction:
    """
    Standard output container produced by any BaseChangeModel implementation.

    The `model_status` property maps the tri-axis state to the single composite
    ModelStatus enum for pipeline/Tool-3 consumers.
    """
    change_mask: np.ndarray
    probability_map: np.ndarray
    model_name: str
    artifact_status: ModelArtifactStatus
    runtime_status: RuntimeStatus
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    decision_tier: DecisionTier = DecisionTier.PROBABLE
    confidence: float = 0.0
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---- Composite status properties ----------------------------------------

    @property
    def model_status(self) -> ModelStatus:
        """
        Derive the composite ModelStatus from tri-axis state.

        Mapping logic (in priority order):
        1. If is_fallback is True                    → HEURISTIC_FALLBACK
        2. If authentic checkpoint + success         → REAL_MODEL
        3. If classical deterministic fixture        → CLASSICAL_ALGORITHM
        4. If unavailable / mock artifact            → UNAVAILABLE
        5. Default                                   → CLASSICAL_ALGORITHM
        """
        if self.is_fallback:
            return ModelStatus.HEURISTIC_FALLBACK
        if self.artifact_status == ModelArtifactStatus.DETERMINISTIC_ALGORITHM:
            return ModelStatus.CLASSICAL_ALGORITHM
        if (
            self.artifact_status in (
                ModelArtifactStatus.REAL_CHECKPOINT,
                ModelArtifactStatus.DOMAIN_ADAPTED_CHECKPOINT,
            )
            and self.runtime_status in (
                RuntimeStatus.INFERENCE_SUCCESS,
                RuntimeStatus.LOADED,
            )
        ):
            return ModelStatus.REAL_MODEL
        if self.artifact_status in (ModelArtifactStatus.UNAVAILABLE, ModelArtifactStatus.MOCK):
            return ModelStatus.UNAVAILABLE
        # TEST_FIXTURE or REAL_CHECKPOINT in non-inference paths → classical treatment
        return ModelStatus.CLASSICAL_ALGORITHM

    @property
    def is_production_ready(self) -> bool:
        """Only true if authentic checkpoint, successful inference, and domain-validated."""
        return (
            self.artifact_status in (
                ModelArtifactStatus.REAL_CHECKPOINT,
                ModelArtifactStatus.DOMAIN_ADAPTED_CHECKPOINT,
            )
            and self.runtime_status == RuntimeStatus.INFERENCE_SUCCESS
            and self.validation_status == ValidationStatus.VALIDATED_ON_TARGET_DOMAIN
        )

    # ---- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_status": self.model_status.value,
            "artifact_status": self.artifact_status.value,
            "runtime_status": self.runtime_status.value,
            "validation_status": self.validation_status.value,
            "decision_tier": self.decision_tier.value,
            "confidence": round(float(self.confidence), 4),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "changed_pixels": int(np.sum(self.change_mask)),
            "total_pixels": int(self.change_mask.size),
            "change_percentage": round(float(np.mean(self.change_mask) * 100.0), 3),
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseChangeModel(ABC):
    """
    Abstract interface for all learned and classical change detection models.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version

    @abstractmethod
    def predict(
        self,
        t1: RasterData,
        t2: RasterData,
        **kwargs: Any,
    ) -> ChangePrediction:
        """Execute change detection between bi-temporal rasters T1 and T2."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model checkpoints and required compute runtimes are ready."""
        raise NotImplementedError
