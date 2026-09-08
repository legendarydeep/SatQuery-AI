"""
backend.app.models.base
=======================
Base class and interfaces for SatQuery AI local neural specialist models.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseAIModel(ABC):
    """
    Abstract Base Class for all SatQuery AI local execution models.
    Supports lifecycle management: load(), predict(), unload(), status().
    """

    def __init__(
        self,
        model_name: str,
        hf_repo_id: Optional[str] = None,
        parameter_count: str = "N/A",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.parameter_count = parameter_count
        self.device = device
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> bool:
        """Loads weights and prepares model for inference."""
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs model inference on provided inputs."""
        pass

    def unload(self) -> None:
        """Releases GPU/CPU memory resources."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        logger.info("Unloaded model: %s", self.model_name)

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "repo_id": self.hf_repo_id,
            "params": self.parameter_count,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "error": self._load_error,
        }
