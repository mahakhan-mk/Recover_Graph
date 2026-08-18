"""Persistence boundary for operational memory."""

from typing import Protocol

from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution


class OperationalMemoryRepository(Protocol):
    def save_failure(self, failure: FailureEpisode) -> None: ...

    def save_resolution(self, resolution: Resolution) -> None: ...

    def save_outcome(self, outcome: Outcome) -> None: ...
