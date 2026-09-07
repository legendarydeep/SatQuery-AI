"""
Tests for Adaptive DAG Planner & Re-Planning Loop (Critiques 1 & 21)
"""

import pytest
from satquery.agent.intent import IntentParser
from satquery.agent.planner import AdaptivePlanner
from satquery.agent.schemas import CrossModalAssessment, ExecutionBudget


def test_initial_dag_generation_and_budgeting():
    intent = IntentParser.parse_query("Compare T1 and T2 deforestation in optical imagery")
    budget = ExecutionBudget(max_tools=3, max_runtime_s=15.0)
    planner = AdaptivePlanner(default_budget=budget)

    dag = planner.plan_initial_dag(intent)
    assert dag is not None
    assert len(dag.nodes) <= budget.max_tools
    assert any("change" in node.selected_tool_id for node in dag.nodes)


def test_adaptive_replanning_on_contradiction():
    intent = IntentParser.parse_query("Analyze flood expansion using Optical and SAR")
    planner = AdaptivePlanner()
    dag = planner.plan_initial_dag(intent)

    # Simulate strong contradiction between optical and SAR
    assessment = CrossModalAssessment(
        category="strong_contradiction",
        relative_difference=0.60,
        absolute_difference_ha=14.5,
        uncertainty_overlap=False,
        modality_profiles={},
        contradiction_penalty=0.25,
        explanation="Severe divergence between optical and SAR flooded area"
    )

    # Planner should re-plan and add supplementary verification node
    replanned_dag = planner.replan_on_disagreement(dag, assessment)
    assert replanned_dag is not None
    assert len(replanned_dag.nodes) > len(dag.nodes)
    assert any("classical_change_detector" in n.selected_tool_id for n in replanned_dag.nodes)


def test_no_replanning_on_concordant_agreement():
    intent = IntentParser.parse_query("Analyze change in optical tiles")
    planner = AdaptivePlanner()
    dag = planner.plan_initial_dag(intent)

    assessment = CrossModalAssessment(
        category="concordant_agreement",
        relative_difference=0.02,
        absolute_difference_ha=0.1,
        uncertainty_overlap=True,
        modality_profiles={},
        contradiction_penalty=0.0,
        explanation="Full agreement"
    )

    replanned = planner.replan_on_disagreement(dag, assessment)
    assert replanned is None
