"""Shared validation helpers for persistence domain contracts."""

from datetime import datetime


def require_non_empty(value: str) -> str:
    """Reject empty identity and relationship-reference strings."""
    if not value.strip():
        raise ValueError("value must be a non-empty string")
    return value


def require_timezone_aware(value: datetime) -> datetime:
    """Require timestamps that carry a usable timezone offset."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value
