"""
satquery.change_detection.evidence_fusion
=========================================
Evidence fusion, cross-corroboration, and discrepancy analysis between learned
deep features (ChangeFormer) and deterministic spectral baselines (NDVI, NDWI, NDBI).
Provides named evidence signals (SUPPORT / DISAGREEMENT / INSUFFICIENT) and auditable
decision tiers without opaque magic-weighted sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from .models.base import DecisionTier


class EvidenceSignal(str, Enum):
    """Named evidence signal categories between learned predictions and physical indices."""
    SUPPORT = "SUPPORT"
    DISAGREEMENT = "DISAGREEMENT"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class EvidenceReport:
    """Quantitative evidence corroboration and discrepancy report."""
    pixel_iou: float
    spectral_support_fraction: float
    spectral_disagreement_fraction: float
    mean_spectral_response: float
    evidence_signal: EvidenceSignal
    alignment_quality: float
    decision_tier: DecisionTier
    diagnostic_reason: str
    classical_supported_fraction: float = 1.0

    # Backward-compatible property aliases
    @property
    def learned_classical_iou(self) -> float:
        return self.pixel_iou

    @property
    def learned_supported_fraction(self) -> float:
        return self.spectral_support_fraction

    @property
    def spectral_contradiction_fraction(self) -> float:
        return self.spectral_disagreement_fraction

    @property
    def evidence_score(self) -> float:
        """Derived metric based on spectral support discounted by disagreement."""
        return float(np.clip(self.spectral_support_fraction * (1.0 - self.spectral_disagreement_fraction), 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_iou": round(self.pixel_iou, 4),
            "spectral_support_fraction": round(self.spectral_support_fraction, 4),
            "spectral_disagreement_fraction": round(self.spectral_disagreement_fraction, 4),
            "mean_spectral_response": round(self.mean_spectral_response, 4),
            "evidence_signal": self.evidence_signal.value,
            "alignment_quality": round(self.alignment_quality, 4),
            "decision_tier": self.decision_tier.value,
            "diagnostic_reason": self.diagnostic_reason,
            # Backward-compatible fields
            "learned_classical_iou": round(self.pixel_iou, 4),
            "learned_supported_fraction": round(self.spectral_support_fraction, 4),
            "classical_supported_fraction": round(self.classical_supported_fraction, 4),
            "spectral_contradiction_fraction": round(self.spectral_disagreement_fraction, 4),
            "evidence_score": round(self.evidence_score, 4),
        }


class EvidenceFusionEngine:
    """
    Fuses multi-spectral indices with learned change probabilities to produce
    auditable decision tiers and named corroboration signals.
    """

    @staticmethod
    def evaluate(
        prob_learned: np.ndarray,
        mask_learned: np.ndarray,
        mask_classical: np.ndarray,
        spectral_diffs: Dict[str, np.ndarray],
        alignment_quality: float = 1.0,
    ) -> EvidenceReport:
        """
        Evaluate named evidence signals: pixel IoU, spectral support fraction,
        spectral disagreement fraction, and mean spectral response.
        """
        # 1. Pixel IoU between learned mask and classical spectral mask
        inter_lc = np.logical_and(mask_learned, mask_classical).sum()
        union_lc = np.logical_or(mask_learned, mask_classical).sum()
        pixel_iou = float(inter_lc / max(1, union_lc))

        # 2. Spectral support: fraction of learned changed pixels corroborated by physical spectral index shifts
        spectral_any = np.zeros_like(mask_learned, dtype=bool)
        spectral_sum = np.zeros_like(mask_learned, dtype=np.float32)
        for diff in spectral_diffs.values():
            thresh = max(0.04, float(np.mean(diff) + 0.5 * np.std(diff)))
            spectral_any |= (diff > thresh)
            spectral_sum += np.abs(diff)

        n_learned = max(1, int(mask_learned.sum()))
        n_classical = max(1, int(mask_classical.sum()))

        support_fraction = float(np.logical_and(mask_learned, spectral_any).sum() / n_learned)
        classical_supported = float(inter_lc / n_classical)
        mean_spectral_resp = float(np.mean(spectral_sum[mask_learned])) if mask_learned.any() else 0.0

        # 3. Spectral disagreement: high learned confidence (P > 0.70) while ALL physical spectral shifts < 0.05
        high_prob = prob_learned > 0.70
        all_spectral_absent = np.ones_like(mask_learned, dtype=bool)
        for diff in spectral_diffs.values():
            all_spectral_absent &= (diff < 0.05)

        disagreement_mask = high_prob & all_spectral_absent & (alignment_quality > 0.80)
        disagreement_fraction = float(disagreement_mask.sum() / max(1, int(high_prob.sum())))

        # 4. Categorical Evidence Signal
        if disagreement_fraction > 0.35 and alignment_quality > 0.80:
            signal = EvidenceSignal.DISAGREEMENT
        elif support_fraction >= 0.50:
            signal = EvidenceSignal.SUPPORT
        else:
            signal = EvidenceSignal.INSUFFICIENT

        # 5. Named Rule-Based Decision Logic
        if signal == EvidenceSignal.DISAGREEMENT:
            tier = DecisionTier.UNCERTAIN
            reason = (
                f"Spectral disagreement ({disagreement_fraction:.1%}): learned features contradict physical "
                f"spectral bands. Flagged for review."
            )
        elif mask_learned.sum() == 0 and mask_classical.sum() == 0:
            tier = DecisionTier.REJECTED
            reason = "Null baseline: no significant surface change detected."
        elif signal == EvidenceSignal.SUPPORT:
            if support_fraction > 0.65 and pixel_iou > 0.35:
                tier = DecisionTier.VERIFIED
                reason = (
                    f"High corroboration: learned features verified by physical spectral indices "
                    f"(support={support_fraction:.1%}, IoU={pixel_iou:.3f})."
                )
            else:
                tier = DecisionTier.PROBABLE
                reason = f"Probable surface change with corroborating spectral evidence ({support_fraction:.1%})."
        else:  # INSUFFICIENT
            if mask_learned.sum() > 0 and pixel_iou > 0.20:
                tier = DecisionTier.PROBABLE
                reason = "Probable change detected by learned model with partial spectral evidence."
            else:
                tier = DecisionTier.REJECTED
                reason = "Weak or transient surface change below evidence significance threshold."

        return EvidenceReport(
            pixel_iou=pixel_iou,
            spectral_support_fraction=support_fraction,
            spectral_disagreement_fraction=disagreement_fraction,
            mean_spectral_response=mean_spectral_resp,
            evidence_signal=signal,
            alignment_quality=alignment_quality,
            decision_tier=tier,
            diagnostic_reason=reason,
            classical_supported_fraction=classical_supported,
        )
