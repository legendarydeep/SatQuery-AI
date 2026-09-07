"""
Agentic Orchestration Schemas (v0.2) — Workstream E & Platform
Defines strongly typed, tamper-evident contracts for:
- QueryIntent & ContextResolution
- MeasurementRecord (Zero-numeric-authority LLM enforcement)
- SpecialistRequest & SpecialistResult (Formal protocol payloads)
- CrossModalAssessment (Multi-scale disagreement arbitration)
- ExecutionBudget & TaskDAG specifications
- TraceStepEvent & RunManifest
"""

from typing import Dict, Any, Optional, List, Tuple, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Intent & Context Resolution
# ---------------------------------------------------------------------------

class ContextResolution(BaseModel):
    """Explicit multi-turn context reference resolution."""
    reference_phrase: str = Field(..., description="e.g. 'same area', 'T2 scene', 'that river'")
    resolved_asset_id: Optional[str] = Field(None, description="Resolved asset identifier")
    resolved_aoi_geojson: Optional[Dict[str, Any]] = Field(None, description="Resolved AOI polygon")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    justification: str = Field(..., description="Audit rationale explaining how context was resolved")


class QueryIntent(BaseModel):
    """Structured intent representation with constraint isolation."""
    schema_version: str = "0.2"
    task_type: Literal[
        "vqa", "scene_description", "grounding", "temporal_change",
        "flood_mapping", "deforestation", "sar_cloud_penetration", "multimodal_fusion"
    ]
    target_entity: str = Field(..., description="Entity of interest, e.g. 'water', 'building', 'forest'")
    requested_outputs: List[Literal["answer", "change_map", "bounding_boxes", "area", "geojson", "report"]]
    modalities: List[Literal["optical", "sar", "multispectral"]]
    temporal: bool = False
    spatial_constraint: Optional[Dict[str, Any]] = None  # GeoJSON bbox or polygon
    confidence_requirement: Literal["standard", "high_precision", "conservative"] = "standard"
    context_resolutions: List[ContextResolution] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Strict Numerical Provenance (Deterministic Measurement Engine)
# ---------------------------------------------------------------------------

class MeasurementRecord(BaseModel):
    """
    Cryptographically traceable numerical measurement.
    LLM synthesizer has ZERO authority to generate or alter these values.
    """
    metric_id: str = Field(..., description="Canonical metric key, e.g. 'area_ha', 'change_pct'")
    value: float = Field(..., description="Deterministic numerical value")
    unit: Literal["ha", "m2", "km2", "%", "count", "meters", "ratio"]
    source_tool: str = Field(..., description="Tool or specialist that computed this number")
    source_run_id: str = Field(..., description="Unique execution run ID")
    raster_hash: str = Field(..., description="SHA-256 hash of the input raster tile")
    computation_version: str = Field("1.0", description="Algorithm version")
    confidence_interval: Optional[Tuple[float, float]] = Field(
        None, description="Optional (lower_bound, upper_bound) uncertainty interval"
    )


# ---------------------------------------------------------------------------
# 3. Specialist Execution Contracts & Status Taxonomy
# ---------------------------------------------------------------------------

class SpecialistRequest(BaseModel):
    """Standardized input payload passed to any registered Specialist."""
    run_id: str
    task_type: str
    query_text: str
    asset_paths: Dict[str, str] = Field(
        ..., description="Map of role/timestamp to file path, e.g. {'t1': 'path.tif', 't2': 'path2.tif'}"
    )
    aoi_geometry: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 10.0


class SpecialistResult(BaseModel):
    """
    Standardized result emitted by any specialist.
    Decouples implementation type from runtime execution status.
    """
    tool_id: str
    implementation_type: Literal[
        "real_model", "domain_adapted_model", "classical_algorithm", "heuristic", "mock"
    ]
    execution_status: Literal[
        "healthy", "degraded", "unavailable", "failed", "timeout"
    ]
    duration_ms: float
    observations: str
    measurements: List[MeasurementRecord] = Field(default_factory=list)
    geojson_geometry: Optional[Dict[str, Any]] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    raw_artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Multi-Scale Cross-Modal Disagreement
# ---------------------------------------------------------------------------

class CrossModalAssessment(BaseModel):
    """
    Multi-scale arbitration between distinct remote sensing modalities (e.g. Optical vs SAR).
    Distinguishes physical sensor complementarity from true contradictions.
    """
    category: Literal[
        "concordant_agreement",
        "cross_modal_complementarity",
        "expected_phenomenological_difference",
        "potential_contradiction",
        "strong_contradiction"
    ]
    relative_difference: float = Field(..., description="Relative difference in [0.0, 1.0]")
    absolute_difference_ha: float = Field(..., description="Absolute difference in hectares")
    uncertainty_overlap: bool = Field(..., description="True if measurement confidence intervals overlap")
    modality_profiles: Dict[str, Any] = Field(default_factory=dict)
    contradiction_penalty: float = Field(0.0, ge=0.0, le=1.0)
    explanation: str


# ---------------------------------------------------------------------------
# 5. Planning, Resource Budgets & Task DAG
# ---------------------------------------------------------------------------

class ExecutionBudget(BaseModel):
    """Hard compute budgets preventing explosive DAG generation."""
    max_tools: int = 5
    max_runtime_s: float = 20.0
    max_gpu_memory_mb: int = 6000
    max_raster_pixels: int = 50_000_000


class TaskNode(BaseModel):
    """Single node in the execution DAG."""
    node_id: str
    capability: str
    selected_tool_id: str
    dependencies: List[str] = Field(default_factory=list, description="IDs of parent nodes")
    estimated_latency_s: float = 1.0
    estimated_compute_cost: float = 1.0
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"


class TaskDAG(BaseModel):
    """Directed Acyclic Graph representing the full execution plan."""
    plan_id: str
    query_intent: QueryIntent
    nodes: List[TaskNode]
    budget: ExecutionBudget
    estimated_total_latency_s: float = 0.0


# ---------------------------------------------------------------------------
# 6. Trace Events & Tamper-Evident Run Manifests
# ---------------------------------------------------------------------------

class TraceStepEvent(BaseModel):
    """Event emitted at each stage of the DAG execution for live SSE streaming."""
    event_id: int = Field(..., description="Sequential integer ID for Last-Event-ID replay")
    run_id: str
    step_index: int
    step_name: str
    tool_id: str
    implementation_type: str
    execution_status: str
    duration_ms: float
    observations: str
    measurements: List[Dict[str, Any]] = Field(default_factory=list)
    why_selected: str = Field(..., description="Justification for why this node was planned/executed")
    timestamp: float


class RunManifest(BaseModel):
    """
    Tamper-evident reproducibility manifest.
    Identifies the exact computational configuration, versions, and hashes of a run.
    """
    manifest_version: str = "1.0"
    run_signature_sha256: str
    environment: Dict[str, str] = Field(
        ..., description="Versions of python, torch, gdal, cuda, and container digest"
    )
    provenance: Dict[str, Any] = Field(
        ..., description="Hashes of input rasters, query, model checkpoints, parameters, and output GeoJSON"
    )
