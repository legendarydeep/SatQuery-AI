"""
Specialist Capability Registry (Workstream E & Platform)
Manages dynamic discovery, capability index, and health monitoring for all registered tools.
Conforms strictly to the formal Specialist Protocol.
"""

from typing import Dict, List, Optional, Any
import logging
from satquery.agent.protocols import Specialist
from satquery.agent.schemas import SpecialistRequest, SpecialistResult

logger = logging.getLogger(__name__)


class RegistryEntry:
    def __init__(
        self,
        specialist: Specialist,
        hardware_tier: str,
        implementation_type: str
    ):
        self.specialist = specialist
        self.hardware_tier = hardware_tier  # "cuda", "mps", "cpu_quantized", "heuristic_pure_python"
        self.implementation_type = implementation_type  # "real_model", "domain_adapted_model", "classical_algorithm", "heuristic", "mock"
        self.health_status = "healthy"  # "healthy", "degraded", "offline"


class SpecialistRegistry:
    """
    Central repository of domain specialists.
    Declares availability; does NOT act as an execution planner.
    """

    def __init__(self):
        self._registry: Dict[str, RegistryEntry] = {}

    def register(
        self,
        specialist: Specialist,
        hardware_tier: str = "cpu_quantized",
        implementation_type: str = "classical_algorithm"
    ) -> None:
        """Registers a specialist instance conforming to the Specialist Protocol."""
        tool_id = specialist.tool_id
        self._registry[tool_id] = RegistryEntry(
            specialist=specialist,
            hardware_tier=hardware_tier,
            implementation_type=implementation_type
        )
        logger.info(
            f"Registered specialist '{tool_id}' (Lead: {specialist.author_lead}, "
            f"Type: {implementation_type}, Tier: {hardware_tier})"
        )

    def get(self, tool_id: str) -> Optional[Specialist]:
        """Retrieves a registered specialist by ID."""
        entry = self._registry.get(tool_id)
        return entry.specialist if entry else None

    def get_entry(self, tool_id: str) -> Optional[RegistryEntry]:
        """Retrieves entry metadata."""
        return self._registry.get(tool_id)

    def find_by_capability(
        self,
        capability: str,
        modality: Optional[str] = None
    ) -> List[Specialist]:
        """
        Discovers all healthy specialists that support the specified capability and modality.
        """
        matching: List[Specialist] = []
        for entry in self._registry.values():
            if entry.health_status == "offline":
                continue
            spec = entry.specialist
            if capability in spec.capabilities():
                if modality is None or modality in spec.modalities():
                    matching.append(spec)
        return matching

    def list_tools(self) -> List[Dict[str, Any]]:
        """Summarizes all registered tools for the API status endpoint."""
        items = []
        for entry in self._registry.values():
            spec = entry.specialist
            items.append({
                "tool_id": spec.tool_id,
                "author_lead": spec.author_lead,
                "capabilities": spec.capabilities(),
                "modalities": spec.modalities(),
                "hardware_tier": entry.hardware_tier,
                "implementation_type": entry.implementation_type,
                "health_status": entry.health_status
            })
        return items

    async def update_health(self, tool_id: str) -> str:
        """Queries specialist for current health status and updates registry."""
        entry = self._registry.get(tool_id)
        if not entry:
            return "offline"
        try:
            status = await entry.specialist.health()
            entry.health_status = status
        except Exception:
            entry.health_status = "offline"
        return entry.health_status


# Global registry singleton
registry = SpecialistRegistry()
