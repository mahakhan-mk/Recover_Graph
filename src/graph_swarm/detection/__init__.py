"""Deterministic classification of agent events into failure episodes."""

from graph_swarm.detection.failure_detector import FailureDetector, detect_failure

__all__ = ["FailureDetector", "detect_failure"]
