from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, observation: str) -> str:
        raise NotImplementedError
