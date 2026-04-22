from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    policy_id: str

    @abstractmethod
    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
