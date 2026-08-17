"""Objective task/action outcome contracts."""

from datetime import datetime

from pydantic import BaseModel


class Outcome(BaseModel):
    action_id: str
    success: bool
    tests_passed: int | None = None
    tests_failed: int | None = None
    exit_code: int | None = None
    observed_at: datetime
