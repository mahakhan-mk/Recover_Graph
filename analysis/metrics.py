"""Primary and secondary Graph Swarm research metrics."""


def repeated_failure_rate(repeated_failures: int, opportunities: int) -> float:
    if opportunities <= 0:
        raise ValueError("opportunities must be positive")
    return repeated_failures / opportunities


def false_advice_rate(false_advice: int, advice_events: int) -> float:
    if advice_events <= 0:
        raise ValueError("advice_events must be positive")
    return false_advice / advice_events
