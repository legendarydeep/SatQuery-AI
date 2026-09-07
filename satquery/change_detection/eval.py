"""
satquery.change_detection.eval
==============================
Quantitative remote-sensing evaluation, geospatial quality assessment, and
ablation comparison suite for SatQuery AI.

Implements all metrics required by the GeoCV charter:
- Pixel-level: Precision, Recall, F1-score, IoU (Jaccard), mIoU, Overall Accuracy.
- Geospatial-level: Polygon IoU, Area Error (m² & %), Centroid Error.
- Model comparison & ablation harness: Classical vs. Learned vs. Ensemble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

from .models.base import ChangePrediction


@dataclass
class PixelMetrics:
    """Standard binary pixel-level segmentation metrics."""
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1_score: float
    iou: float
    mean_iou: float
    overall_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1_score, 4),
            "iou": round(self.iou, 4),
            "miou": round(self.mean_iou, 4),
            "accuracy": round(self.overall_accuracy, 4),
        }


@dataclass
class GeospatialMetrics:
    """Geospatial validation and robust boundary quality metrics."""
    polygon_iou: float
    area_error_m2: float
    area_error_pct: float
    predicted_area_m2: float
    true_area_m2: float
    boundary_f1: float = 1.0
    hausdorff_95: float = 0.0
    centroid_error_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_iou": round(self.polygon_iou, 4),
            "area_error_m2": round(self.area_error_m2, 2),
            "area_error_pct": round(self.area_error_pct, 2),
            "predicted_area_m2": round(self.predicted_area_m2, 2),
            "true_area_m2": round(self.true_area_m2, 2),
            "boundary_f1": round(self.boundary_f1, 4),
            "hausdorff_95": round(self.hausdorff_95, 2),
            "centroid_error_m": round(self.centroid_error_m, 2),
        }


def compute_pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> PixelMetrics:
    """
    Compute Precision, Recall, F1, IoU, mIoU, and Overall Accuracy
    comparing a predicted binary mask against ground truth.
    """
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(1, total)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    denom_f1 = precision + recall
    f1 = (2.0 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    denom_iou = tp + fp + fn
    iou_change = tp / max(1, denom_iou) if denom_iou > 0 else (1.0 if fp == 0 and fn == 0 else 0.0)

    # Background IoU
    denom_bg = tn + fp + fn
    iou_bg = tn / max(1, denom_bg) if denom_bg > 0 else 1.0

    mean_iou = (iou_change + iou_bg) / 2.0

    return PixelMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        iou=iou_change,
        mean_iou=mean_iou,
        overall_accuracy=accuracy,
    )


def compute_geospatial_metrics(
    pred_geojson: dict[str, Any],
    gt_geojson: dict[str, Any],
    pixel_area_m2: float = 100.0,
) -> GeospatialMetrics:
    """
    Compute vector-based Polygon IoU and area discrepancy between
    predicted GeoJSON and ground-truth GeoJSON.
    """
    def _extract_poly(fc: dict[str, Any]):
        geoms = []
        for feat in fc.get("features", []):
            try:
                g = shape(feat.get("geometry", {}))
                if g.is_valid and not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            return None
        return unary_union(geoms)

    poly_pred = _extract_poly(pred_geojson)
    poly_gt = _extract_poly(gt_geojson)

    if poly_pred is None and poly_gt is None:
        return GeospatialMetrics(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if poly_pred is None:
        area_gt = poly_gt.area
        return GeospatialMetrics(0.0, float(area_gt), 100.0, 0.0, float(area_gt), 0.0, 999.0, 999.0)
    if poly_gt is None:
        area_pred = poly_pred.area
        return GeospatialMetrics(0.0, float(area_pred), 100.0, float(area_pred), 0.0, 0.0, 999.0, 999.0)

    intersection = poly_pred.intersection(poly_gt).area
    union = poly_pred.union(poly_gt).area
    poly_iou = (intersection / union) if union > 0 else 0.0

    pred_area = float(poly_pred.area)
    gt_area = float(poly_gt.area)
    area_diff = abs(pred_area - gt_area)
    area_pct = (area_diff / max(1e-6, gt_area)) * 100.0

    # Centroid shift
    c_pred = poly_pred.centroid
    c_gt = poly_gt.centroid
    centroid_dist = float(c_pred.distance(c_gt))

    # Boundary Hausdorff distance approximation
    try:
        b_pred = poly_pred.boundary
        b_gt = poly_gt.boundary
        h_dist = float(b_pred.hausdorff_distance(b_gt))
    except Exception:
        h_dist = centroid_dist

    # Boundary F1: buffer precision and recall
    try:
        buf_size = max(0.0001, math.sqrt(pred_area) * 0.02)
        buf_gt = poly_gt.boundary.buffer(buf_size)
        buf_pred = poly_pred.boundary.buffer(buf_size)
        prec = float(poly_pred.boundary.intersection(buf_gt).length / max(1e-6, poly_pred.boundary.length))
        rec = float(poly_gt.boundary.intersection(buf_pred).length / max(1e-6, poly_gt.boundary.length))
        bf1 = float(2 * prec * rec / max(1e-6, prec + rec))
    except Exception:
        bf1 = poly_iou

    return GeospatialMetrics(
        polygon_iou=poly_iou,
        area_error_m2=area_diff,
        area_error_pct=area_pct,
        predicted_area_m2=pred_area,
        true_area_m2=gt_area,
        boundary_f1=min(1.0, max(0.0, bf1)),
        hausdorff_95=h_dist,
        centroid_error_m=centroid_dist,
    )


class ModelAblationComparator:
    """
    Compares predictions from multiple models or ablation conditions
    against a ground truth mask and against each other.
    """

    def __init__(self, ground_truth_mask: np.ndarray):
        self.gt_mask = ground_truth_mask

    def evaluate_predictions(
        self,
        predictions: Dict[str, ChangePrediction],
    ) -> Dict[str, Any]:
        """
        Evaluate and rank all model predictions.
        """
        results: Dict[str, Any] = {}
        for name, pred in predictions.items():
            pm = compute_pixel_metrics(pred.change_mask, self.gt_mask)
            results[name] = {
                "model_status": pred.model_status.value,
                "confidence": pred.confidence,
                "metrics": pm.to_dict(),
                "provenance": pred.provenance,
            }

        # Cross-model agreement (IoU between predictions)
        agreements: Dict[str, float] = {}
        keys = list(predictions.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m1, m2 = keys[i], keys[j]
                mask1 = predictions[m1].change_mask
                mask2 = predictions[m2].change_mask
                inter = np.logical_and(mask1, mask2).sum()
                uni = np.logical_or(mask1, mask2).sum()
                iou_pair = float(inter / max(1, uni))
                agreements[f"{m1}_vs_{m2}"] = round(iou_pair, 4)

        return {
            "model_evaluations": results,
            "cross_model_agreement": agreements,
        }
