"""
PDF Dossier Export (Workstream E / Critique 10 - P2)
Exports a mission result or EvaluationRecord to a structured PDF dossier.
Falls back to JSON if fpdf2 is not installed.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import json
import io
import datetime


def _try_fpdf():
    try:
        from fpdf import FPDF
        return FPDF
    except ImportError:
        return None


def build_pdf_dossier(
    mission_id: str,
    query: str,
    answer: str,
    measurements: Dict[str, float],
    confidence: float,
    confidence_breakdown: Optional[Dict[str, float]] = None,
    run_id: Optional[str] = None,
) -> bytes:
    """
    Returns PDF bytes of the mission dossier, or JSON bytes if fpdf2 is absent.
    """
    FPDF = _try_fpdf()
    if FPDF is None:
        # Graceful degradation: return JSON dossier
        doc = {
            "mission_id": mission_id,
            "run_id": run_id,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "measurements": measurements,
            "confidence": confidence,
            "confidence_breakdown": confidence_breakdown or {},
        }
        return json.dumps(doc, indent=2).encode("utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "SatQuery AI - Mission Dossier", ln=True, align="C")

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Mission ID : {mission_id}", ln=True)
    if run_id:
        pdf.cell(0, 6, f"Run ID     : {run_id}", ln=True)
    pdf.cell(0, 6, f"Generated  : {datetime.datetime.now(datetime.timezone.utc).isoformat()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Query", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, query)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Answer", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, answer)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Composite Confidence: {confidence:.3f}", ln=True)

    if confidence_breakdown:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Confidence Breakdown", ln=True)
        pdf.set_font("Helvetica", size=10)
        for factor, score in confidence_breakdown.items():
            bar = "#" * int(float(score) * 20)
            pdf.cell(0, 6, f"  {factor:<30} {float(score):.2f}  {bar}", ln=True)
    pdf.ln(3)

    if measurements:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Key Measurements", ln=True)
        pdf.set_font("Helvetica", size=10)
        for k, v in measurements.items():
            unit = "ha" if "ha" in k else "m2" if "m2" in k else "%" if "pct" in k else ""
            pdf.cell(0, 6, f"  {k}: {v} {unit}".rstrip(), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Produced by SatQuery AI (ISRO/SAC SIH26167) - Team SIH059", align="C")

    return pdf.output(dest="S").encode("latin-1") if isinstance(pdf.output(dest="S"), str)         else bytes(pdf.output(dest="S"))
