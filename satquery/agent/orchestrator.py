"""
Top-Level Agent Orchestrator (Workstream E & Platform)
Coordinates Intent Parsing -> Adaptive Planning -> Asynchronous DAG Execution ->
Evidence Fusion -> Guardrail Verification -> Tamper-Evident Run Manifest Assembly.
"""

import logging
from typing import Dict, Any, List, Optional
from satquery.agent.intent import IntentParser
from satquery.agent.planner import AdaptivePlanner
from satquery.agent.executor import TaskExecutor
from satquery.agent.trace import trace_store, ReproducibilityManager
from satquery.evaluation.guardrails import NumericalGuardrail

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Main orchestration engine for SatQuery AI.
    Converts natural language queries and satellite rasters into honest, verifiable answers.
    """

    def __init__(self):
        self.intent_parser = IntentParser()
        self.planner = AdaptivePlanner()
        self.guardrail = NumericalGuardrail(tolerance=0.05)

    async def run_query(
        self,
        query: str,
        asset_paths: Dict[str, str],
        mission_id: Optional[str] = None,
        prior_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end agentic workflow with adaptive re-planning and guardrail verification.
        """
        # 1. Parse Intent & resolve context
        intent = self.intent_parser.parse_query(
            query=query,
            active_assets=asset_paths,
            prior_context=prior_context
        )

        # 2. Plan initial execution DAG
        dag = self.planner.plan_initial_dag(intent)

        # 3. Execute DAG with adaptive re-planning
        executor = TaskExecutor(planner=self.planner)
        exec_output = await executor.execute_dag(
            dag=dag,
            asset_paths=asset_paths,
            query_text=query
        )

        run_id = exec_output["run_id"]
        fused = exec_output["fused_result"]
        trace_events = exec_output["trace_events"]

        # Store trace events in durable TraceStore
        for ev in executor._trace_events:
            trace_store.append_event(run_id, ev)

        # 4. Strict Numerical Protection & Guardrail Check (Critique 5)
        deterministic_nums: Dict[str, float] = {}
        for m in fused["measurements"]:
            deterministic_nums[m["metric_id"]] = float(m["value"])

        # Check synthesized text against ground truth measurements
        passed, violations, rectified_text = self.guardrail.verify(
            text=fused["synthesized_answer"],
            deterministic_numbers=deterministic_nums,
            geojson_geometry=fused.get("geojson_geometry")
        )

        # 5. Assemble tamper-evident run manifest (Critique 6)
        tool_versions = {
            ev["tool_id"]: ev.get("implementation_type", "v1.0") for ev in trace_events
        }
        manifest = ReproducibilityManager.generate_manifest(
            run_id=run_id,
            query=query,
            planner_version=intent.schema_version,
            input_file_paths=list(asset_paths.values()),
            tool_versions=tool_versions,
            output_data={"answer": rectified_text}
        )

        return {
            "run_id": run_id,
            "mission_id": mission_id,
            "status": exec_output["execution_status"],
            "query_intent": intent.model_dump(),
            "answer": rectified_text,
            "measurements": fused["measurements"],
            "geojson_geometry": fused["geojson_geometry"],
            "composite_confidence": fused["composite_confidence"],
            "confidence_breakdown": fused["confidence_breakdown"],
            "cross_modal_assessment": fused["cross_modal_assessment"],
            "guardrail_verification": {
                "passed": passed,
                "violations": [v.to_dict() for v in violations]
            },
            "run_manifest": manifest.model_dump(),
            "trace_events": trace_events
        }


# Global orchestrator instance
orchestrator = AgentOrchestrator()
