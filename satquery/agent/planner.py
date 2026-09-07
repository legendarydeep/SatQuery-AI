"""
Adaptive Capability-Based DAG Planner (Critiques 1 & 21)
Constructs, budgets, and re-plans execution graphs based on discovered capabilities.
Implements the core adaptive loop: Plan -> Execute -> Evaluate Evidence -> Re-Plan.
"""

from typing import List, Dict, Any, Optional
import uuid
import logging
from satquery.agent.schemas import (
    QueryIntent, TaskDAG, TaskNode, ExecutionBudget, CrossModalAssessment
)
from satquery.agent.protocols import Specialist
from satquery.agent.registry import registry

logger = logging.getLogger(__name__)


class AdaptivePlanner:
    """
    Constructs executable TaskDAG from QueryIntent and registered specialists.
    Supports dynamic re-planning when initial evidence is contradictory or insufficient.
    """

    def __init__(self, default_budget: Optional[ExecutionBudget] = None):
        self.budget = default_budget or ExecutionBudget()

    def plan_initial_dag(
        self,
        intent: QueryIntent,
        available_specialists: Optional[List[Specialist]] = None
    ) -> TaskDAG:
        """
        Builds initial task DAG based on declared capabilities matching the intent.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        nodes: List[TaskNode] = []

        # Find candidate tools from registry
        specs = available_specialists or registry.find_by_capability(intent.task_type)

        if not specs:
            # Check fallback capability (e.g. temporal_change as fallback for deforestation)
            if intent.task_type in ["deforestation", "flood_mapping"]:
                specs = registry.find_by_capability("temporal_change")

        if not specs:
            # Fallback to general VQA or classical detector
            specs = registry.find_by_capability("vqa") or registry.find_by_capability("temporal_change")

        # Construct primary nodes within budget
        tool_count = 0
        for spec in specs:
            if tool_count >= self.budget.max_tools:
                break

            node = TaskNode(
                node_id=f"node_{len(nodes)+1}_{spec.tool_id}",
                capability=intent.task_type,
                selected_tool_id=spec.tool_id,
                dependencies=[],
                estimated_latency_s=2.0,
                estimated_compute_cost=1.0,
                status="pending"
            )
            nodes.append(node)
            tool_count += 1

        # Multi-modal queries benefit from parallel specialist runs
        if "sar" in intent.modalities and "optical" in intent.modalities:
            sar_specs = registry.find_by_capability("sar_cloud_penetration")
            for sar_spec in sar_specs:
                if tool_count >= self.budget.max_tools:
                    break
                if sar_spec.tool_id not in [n.selected_tool_id for n in nodes]:
                    nodes.append(TaskNode(
                        node_id=f"node_{len(nodes)+1}_{sar_spec.tool_id}",
                        capability="sar_cloud_penetration",
                        selected_tool_id=sar_spec.tool_id,
                        dependencies=[],
                        estimated_latency_s=1.5,
                        estimated_compute_cost=0.8,
                        status="pending"
                    ))
                    tool_count += 1

        est_latency = sum(n.estimated_latency_s for n in nodes)
        return TaskDAG(
            plan_id=plan_id,
            query_intent=intent,
            nodes=nodes,
            budget=self.budget,
            estimated_total_latency_s=est_latency
        )

    def replan_on_disagreement(
        self,
        current_dag: TaskDAG,
        assessment: CrossModalAssessment
    ) -> Optional[TaskDAG]:
        """
        Adaptive Re-Planning Loop:
        If evidence evaluation detects potential contradiction or high ambiguity,
        the planner schedules supplementary deterministic verification tools.
        """
        if assessment.category not in ["potential_contradiction", "strong_contradiction"]:
            # Evidence is consistent or complementary — no re-plan required
            return None

        # Check remaining tool budget
        current_tool_ids = {n.selected_tool_id for n in current_dag.nodes}
        if len(current_dag.nodes) >= self.budget.max_tools:
            logger.warning("Re-planning aborted: Maximum tool budget reached.")
            return None

        new_nodes = list(current_dag.nodes)
        
        # Schedule deterministic classical change detector or supplementary sensor if not already executed
        if "classical_change_detector" not in current_tool_ids:
            supp_spec = registry.get("classical_change_detector")
            if supp_spec:
                new_nodes.append(TaskNode(
                    node_id=f"replan_node_{len(new_nodes)+1}_{supp_spec.tool_id}",
                    capability="temporal_change",
                    selected_tool_id=supp_spec.tool_id,
                    dependencies=[n.node_id for n in current_dag.nodes],
                    estimated_latency_s=1.0,
                    estimated_compute_cost=0.5,
                    status="pending"
                ))
        else:
            # If classical detector already in DAG, schedule supplementary cross-modal tool (e.g. SAR)
            for supp_tool_id in ["sar_microwave_analyzer", "geochat_vlm"]:
                if supp_tool_id not in current_tool_ids and registry.get(supp_tool_id):
                    new_nodes.append(TaskNode(
                        node_id=f"replan_node_{len(new_nodes)+1}_{supp_tool_id}",
                        capability="sar_cloud_penetration" if "sar" in supp_tool_id else "vqa",
                        selected_tool_id=supp_tool_id,
                        dependencies=[n.node_id for n in current_dag.nodes],
                        estimated_latency_s=1.2,
                        estimated_compute_cost=0.6,
                        status="pending"
                    ))
                    break

        if len(new_nodes) == len(current_dag.nodes):
            return None

        logger.info(f"Adaptive re-planning triggered: added {len(new_nodes)-len(current_dag.nodes)} supplementary tool(s).")
        return TaskDAG(
            plan_id=f"{current_dag.plan_id}_replan",
            query_intent=current_dag.query_intent,
            nodes=new_nodes,
            budget=self.budget,
            estimated_total_latency_s=sum(n.estimated_latency_s for n in new_nodes)
        )
