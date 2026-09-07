"""
Tests for Specialist Protocol & Registry (Critiques 16, 17, 18)
"""

import pytest
from satquery.agent.protocols import Specialist, BaseSpecialist
from satquery.agent.schemas import SpecialistRequest, SpecialistResult
from satquery.agent.registry import SpecialistRegistry


class DummyValidSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(
            tool_id="dummy_tool",
            author_lead="omkar",
            supported_capabilities=["vqa", "scene_description"],
            supported_modalities=["optical"]
        )

    async def execute(self, request: SpecialistRequest) -> SpecialistResult:
        return SpecialistResult(
            tool_id=self.tool_id,
            implementation_type="mock",
            execution_status="healthy",
            duration_ms=10.0,
            observations="Dummy execution passed",
            measurements=[],
            confidence_score=0.90
        )


def test_specialist_protocol_conformance():
    spec = DummyValidSpecialist()
    assert isinstance(spec, Specialist)
    assert spec.tool_id == "dummy_tool"
    assert spec.author_lead == "omkar"
    assert "vqa" in spec.capabilities()
    assert "optical" in spec.modalities()


def test_registry_registration_and_lookup():
    reg = SpecialistRegistry()
    spec = DummyValidSpecialist()
    reg.register(spec, hardware_tier="cpu_quantized", implementation_type="mock")

    retrieved = reg.get("dummy_tool")
    assert retrieved is not None
    assert retrieved.tool_id == "dummy_tool"

    found = reg.find_by_capability("vqa")
    assert len(found) == 1
    assert found[0].tool_id == "dummy_tool"

    # Capability not supported
    none_found = reg.find_by_capability("sar_cloud_penetration")
    assert len(none_found) == 0


import asyncio

def test_registry_health_check():
    reg = SpecialistRegistry()
    spec = DummyValidSpecialist()
    reg.register(spec)
    health = asyncio.run(reg.update_health("dummy_tool"))
    assert health == "healthy"
