"""
Tests for Capability-Specific Fallback Policy (Critiques 7 & 8)
Ensures no unrelated algorithms are substituted merely to force an answer.
"""

import pytest
from satquery.agent.fallback import FallbackPolicyEngine


def test_temporal_change_fallback_chain():
    fb = FallbackPolicyEngine.get_fallback_tool("temporal_change", "changeformer_v2")
    assert fb == "classical_change_detector"


def test_scene_description_refuses_unrelated_fallback():
    # If compact_cpu_vlm also fails, it must NOT fall back to spectral differencing!
    fb = FallbackPolicyEngine.get_fallback_tool("scene_description", "compact_cpu_vlm")
    assert fb is None

    result = FallbackPolicyEngine.create_unavailable_result(
        capability="scene_description",
        failed_tool_id="geochat_vlm",
        reason="Model timed out and no classical substitute is valid."
    )
    assert result.execution_status == "unavailable"
    assert "No validated fallback exists" in result.warnings[0]
