"""
Asynchronous DAG Task Executor (Critiques 1, 7, 8, 9, 20)
Executes specialist tasks in parallel, manages retry/fallback, records trace events,
and coordinates the adaptive re-planning feedback loop.
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from satquery.agent.schemas import (
    TaskDAG, TaskNode, SpecialistRequest, SpecialistResult, TraceStepEvent, RunManifest
)
from satquery.agent.registry import registry
from satquery.agent.fallback import FallbackPolicyEngine
from satquery.agent.fusion import (
    CrossModalArbitrationEngine, CalibratedConfidenceEngine, EvidenceFusionEngine
)
from satquery.agent.planner import AdaptivePlanner

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Asynchronously executes a TaskDAG across registered Specialists.
    """

    def __init__(self, planner: Optional[AdaptivePlanner] = None):
        self.planner = planner or AdaptivePlanner()
        self._trace_events: List[TraceStepEvent] = []
        self._event_counter: int = 0

    def _record_trace_event(
        self,
        run_id: str,
        step_name: str,
        tool_id: str,
        impl_type: str,
        status: str,
        duration_ms: float,
        obs: str,
        measurements: List[Dict[str, Any]],
        why_selected: str
    ) -> TraceStepEvent:
        self._event_counter += 1
        event = TraceStepEvent(
            event_id=self._event_counter,
            run_id=run_id,
            step_index=len(self._trace_events) + 1,
            step_name=step_name,
            tool_id=tool_id,
            implementation_type=impl_type,
            execution_status=status,
            duration_ms=round(duration_ms, 2),
            observations=obs,
            measurements=measurements,
            why_selected=why_selected,
            timestamp=time.time()
        )
        self._trace_events.append(event)
        return event

    async def _execute_single_node(
        self,
        node: TaskNode,
        request: SpecialistRequest
    ) -> SpecialistResult:
        """Executes a single specialist node with capability-specific fallback."""
        tool = registry.get(node.selected_tool_id)
        entry = registry.get_entry(node.selected_tool_id)
        impl_type = entry.implementation_type if entry else "classical_algorithm"

        if not tool:
            # Check for capability fallback
            fallback_tool_id = FallbackPolicyEngine.get_fallback_tool(node.capability, node.selected_tool_id)
            if fallback_tool_id and registry.get(fallback_tool_id):
                tool = registry.get(fallback_tool_id)
                impl_type = "heuristic"
                logger.warning(f"Primary tool '{node.selected_tool_id}' missing; degraded to fallback '{fallback_tool_id}'.")
            else:
                return FallbackPolicyEngine.create_unavailable_result(
                    node.capability,
                    node.selected_tool_id,
                    "Tool not registered and no validated fallback exists."
                )

        t0 = time.time()
        try:
            # Enforce timeout budget
            result = await asyncio.wait_for(
                tool.execute(request),
                timeout=request.timeout_s
            )
            node.status = "completed"
            return result
        except asyncio.TimeoutError:
            duration = (time.time() - t0) * 1000.0
            logger.warning(f"Specialist '{tool.tool_id}' timed out after {request.timeout_s}s.")
            # Trigger fallback policy
            fallback_tool_id = FallbackPolicyEngine.get_fallback_tool(node.capability, tool.tool_id)
            if fallback_tool_id and registry.get(fallback_tool_id):
                fallback_tool = registry.get(fallback_tool_id)
                logger.info(f"Invoking fallback tool '{fallback_tool_id}'...")
                res = await fallback_tool.execute(request)
                res.warnings.append(f"Primary tool '{tool.tool_id}' timed out; executed fallback '{fallback_tool_id}'.")
                node.status = "completed"
                return res
            else:
                node.status = "failed"
                return FallbackPolicyEngine.create_unavailable_result(
                    node.capability,
                    tool.tool_id,
                    f"Timed out after {request.timeout_s}s and no validated fallback exists."
                )
        except Exception as e:
            duration = (time.time() - t0) * 1000.0
            logger.error(f"Specialist '{tool.tool_id}' failed: {e}")
            fallback_tool_id = FallbackPolicyEngine.get_fallback_tool(node.capability, tool.tool_id)
            if fallback_tool_id and registry.get(fallback_tool_id):
                fallback_tool = registry.get(fallback_tool_id)
                res = await fallback_tool.execute(request)
                res.warnings.append(f"Primary tool '{tool.tool_id}' threw exception: {e}; used fallback '{fallback_tool_id}'.")
                node.status = "completed"
                return res
            node.status = "failed"
            return FallbackPolicyEngine.create_unavailable_result(
                node.capability,
                tool.tool_id,
                f"Execution exception: {e}"
            )

    async def execute_dag(
        self,
        dag: TaskDAG,
        asset_paths: Dict[str, str],
        query_text: str
    ) -> Dict[str, Any]:
        """
        Executes TaskDAG, evaluates evidence, invokes re-planning if contradictory,
        and returns fused results with live trace events.
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        specialist_results: List[SpecialistResult] = []

        # 1. Dispatch initial nodes concurrently
        tasks = []
        for node in dag.nodes:
            req = SpecialistRequest(
                run_id=run_id,
                task_type=node.capability,
                query_text=query_text,
                asset_paths=asset_paths,
                timeout_s=node.estimated_latency_s * 3.0
            )
            tasks.append(self._execute_single_node(node, req))

        results = await asyncio.gather(*tasks)
        for node, res in zip(dag.nodes, results):
            specialist_results.append(res)
            self._record_trace_event(
                run_id=run_id,
                step_name=f"execute_{node.capability}",
                tool_id=res.tool_id,
                impl_type=res.implementation_type,
                status=res.execution_status,
                duration_ms=res.duration_ms,
                obs=res.observations,
                measurements=[m.model_dump() for m in res.measurements],
                why_selected=f"Planned for capability '{node.capability}'"
            )

        # 2. Evaluate evidence and detect cross-modal disagreement
        opt_res = next((r for r in specialist_results if "optical" in registry.get(r.tool_id).modalities()), None) if specialist_results else None
        sar_res = next((r for r in specialist_results if "sar" in registry.get(r.tool_id).modalities()), None) if specialist_results else None

        assessment = CrossModalArbitrationEngine.arbitrate(opt_res, sar_res)
        self._record_trace_event(
            run_id=run_id,
            step_name="cross_modal_arbitration",
            tool_id="arbitration_engine",
            impl_type="classical_algorithm",
            status="healthy",
            duration_ms=5.0,
            obs=assessment.explanation,
            measurements=[],
            why_selected="Evaluated sensor agreement and physical complementarity."
        )

        # 3. Adaptive Re-Planning Loop (Critique 1)
        replan_dag = self.planner.replan_on_disagreement(dag, assessment)
        if replan_dag:
            self._record_trace_event(
                run_id=run_id,
                step_name="adaptive_replanning",
                tool_id="adaptive_planner",
                impl_type="classical_algorithm",
                status="healthy",
                duration_ms=8.0,
                obs=f"Contradiction detected ({assessment.category}). Scheduled supplementary verification tool.",
                measurements=[],
                why_selected="Disagreement required supplementary deterministic validation."
            )
            # Execute newly added nodes
            new_nodes = replan_dag.nodes[len(dag.nodes):]
            for node in new_nodes:
                req = SpecialistRequest(
                    run_id=run_id,
                    task_type=node.capability,
                    query_text=query_text,
                    asset_paths=asset_paths
                )
                supp_res = await self._execute_single_node(node, req)
                specialist_results.append(supp_res)
                self._record_trace_event(
                    run_id=run_id,
                    step_name=f"execute_supplementary_{node.capability}",
                    tool_id=supp_res.tool_id,
                    impl_type=supp_res.implementation_type,
                    status=supp_res.execution_status,
                    duration_ms=supp_res.duration_ms,
                    obs=supp_res.observations,
                    measurements=[m.model_dump() for m in supp_res.measurements],
                    why_selected="Adaptive re-plan supplementary execution"
                )
            # Re-arbitrate with supplementary evidence
            assessment = CrossModalArbitrationEngine.arbitrate(specialist_results[0], specialist_results[-1])

        # 4. Calibrate Decomposed Confidence
        composite_conf, breakdown = CalibratedConfidenceEngine.calculate(specialist_results, assessment)

        # 5. Fuse final evidence
        fused = EvidenceFusionEngine.fuse(specialist_results, assessment, breakdown, composite_conf)

        return {
            "run_id": run_id,
            "fused_result": fused,
            "trace_events": [ev.model_dump() for ev in self._trace_events],
            "execution_status": "completed" if any(r.execution_status == "healthy" for r in specialist_results) else "failed"
        }
