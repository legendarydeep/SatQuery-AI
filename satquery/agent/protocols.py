"""
Specialist Protocol & Base Class (Workstream E & Platform)
Defines the formal typed interface that every domain specialist must conform to.
Replaces loose 'endpoint_or_callable: Any' with runtime-enforced contract.
"""

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
from satquery.agent.schemas import SpecialistRequest, SpecialistResult


@runtime_checkable
class Specialist(Protocol):
    """Formal protocol for all SatQuery AI domain specialists."""

    @property
    def tool_id(self) -> str:
        """Unique identifier of the specialist tool."""
        ...

    @property
    def author_lead(self) -> str:
        """Name of the owning lead (omkar, aarfa, moiz, kartika, vikram)."""
        ...

    def capabilities(self) -> List[str]:
        """List of supported capability tags (e.g. ['temporal_change', 'grounding'])."""
        ...

    def modalities(self) -> List[str]:
        """List of supported sensor modalities (e.g. ['optical', 'sar'])."""
        ...

    async def health(self) -> str:
        """Returns 'healthy', 'degraded', or 'offline'."""
        ...

    async def execute(self, request: SpecialistRequest) -> SpecialistResult:
        """Executes domain-specific inference or computation."""
        ...


class BaseSpecialist:
    """Convenience base class providing boilerplate for specialists."""

    def __init__(
        self,
        tool_id: str,
        author_lead: str,
        supported_capabilities: List[str],
        supported_modalities: List[str]
    ):
        self._tool_id = tool_id
        self._author_lead = author_lead
        self._capabilities = supported_capabilities
        self._modalities = supported_modalities

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @property
    def author_lead(self) -> str:
        return self._author_lead

    def capabilities(self) -> List[str]:
        return list(self._capabilities)

    def modalities(self) -> List[str]:
        return list(self._modalities)

    async def health(self) -> str:
        return "healthy"

    async def execute(self, request: SpecialistRequest) -> SpecialistResult:
        raise NotImplementedError("Specialist subclass must implement execute()")
