"""
satquery.change_detection.thresholding
======================================
Configurable and auditable ThresholdSelector. Guards against Otsu breakdown
on predominantly unimodal remote sensing change probability distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configurable hyperparameters for threshold selection."""
    method: str = "auto"                    # "auto", "otsu", "adaptive_tail", "fixed"
    bimodality_threshold: float = 0.60       # Minimum bimodality separation to justify Otsu
    valley_threshold: float = 0.40           # Maximum valley-to-peak depth
    tail_percentile: float = 98.0            # Tail percentile for unimodal background
    fixed_cutoff: float = 0.50               # Calibrated probability cutoff


@dataclass
class ThresholdReport:
    """Audit report for threshold selection."""
    method_selected: str
    threshold_value: float
    is_bimodal: bool
    bimodality_score: float
    valley_depth: float
    background_ratio: float
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_selected": self.method_selected,
            "threshold_value": round(self.threshold_value, 5),
            "is_bimodal": self.is_bimodal,
            "bimodality_score": round(self.bimodality_score, 4),
            "valley_depth": round(self.valley_depth, 4),
            "background_ratio": round(self.background_ratio, 4),
            "selection_reason": self.selection_reason,
        }


class ThresholdSelector:
    """
    Selects optimal segmentation cutoff based on histogram distribution diagnostics.
    """

    @staticmethod
    def select_threshold(
        prob_or_diff: np.ndarray,
        config: ThresholdConfig | None = None,
    ) -> Tuple[np.ndarray, ThresholdReport]:
        cfg = config or ThresholdConfig()
        flat = prob_or_diff.flatten().astype(np.float32)
        flat = np.nan_to_num(flat, nan=0.0)

        # 1. Compute distribution diagnostics
        hist, bin_edges = np.histogram(flat, bins=256, range=(0.0, float(np.max(flat)) + 1e-6))
        total_px = max(1, flat.size)
        weights = hist / total_px

        # Inter-class variance calculation (Otsu metric)
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        weight1 = np.cumsum(weights)
        weight2 = 1.0 - weight1
        weight2[weight2 <= 0] = 1e-6

        mean1 = np.cumsum(weights * bin_mids) / np.maximum(weight1, 1e-6)
        mean2 = (np.cumsum((weights * bin_mids)[::-1]) / np.maximum(weight2[::-1], 1e-6))[::-1]

        variance = weight1 * weight2 * (mean1 - mean2) ** 2
        otsu_idx = int(np.argmax(variance))
        otsu_thresh = float(bin_mids[otsu_idx])

        total_variance = float(np.var(flat)) + 1e-8
        bimodality_score = float(variance[otsu_idx] / total_variance)

        # Valley depth
        peak1 = float(np.max(hist[:otsu_idx])) if otsu_idx > 0 else 1.0
        peak2 = float(np.max(hist[otsu_idx:])) if otsu_idx < len(hist) else 1.0
        valley = float(hist[otsu_idx])
        valley_depth = float(valley / max(1.0, min(peak1, peak2)))

        is_bimodal = (bimodality_score >= cfg.bimodality_threshold) and (valley_depth <= cfg.valley_threshold)
        bg_ratio = float(np.sum(flat < otsu_thresh) / total_px)

        # 2. Strategy Selection
        if cfg.method == "fixed":
            chosen_thresh = cfg.fixed_cutoff
            chosen_method = "fixed"
            reason = "Explicit fixed cutoff applied per configuration."
        elif cfg.method == "otsu" or (cfg.method == "auto" and is_bimodal):
            chosen_thresh = otsu_thresh
            chosen_method = "otsu"
            reason = f"Bimodal structure confirmed (Score {bimodality_score:.2f} >= {cfg.bimodality_threshold})."
        else:
            # Unimodal distribution (e.g. 95% unchanged) -> use adaptive tail percentile
            chosen_thresh = float(np.percentile(flat, cfg.tail_percentile))
            chosen_method = "adaptive_tail"
            reason = (
                f"Unimodal distribution (Bimodality {bimodality_score:.2f} < {cfg.bimodality_threshold}, "
                f"Background {bg_ratio:.1%}). Adaptive {cfg.tail_percentile}th percentile applied."
            )

        binary_mask = (prob_or_diff >= chosen_thresh)

        report = ThresholdReport(
            method_selected=chosen_method,
            threshold_value=chosen_thresh,
            is_bimodal=is_bimodal,
            bimodality_score=bimodality_score,
            valley_depth=valley_depth,
            background_ratio=bg_ratio,
            selection_reason=reason,
        )

        return binary_mask, report
