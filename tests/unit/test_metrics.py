import pytest

from analysis.metrics import false_advice_rate, repeated_failure_rate


def test_repeated_failure_rate() -> None:
    assert repeated_failure_rate(2, 8) == 0.25


def test_false_advice_rate() -> None:
    assert false_advice_rate(1, 4) == 0.25


def test_rates_require_denominator() -> None:
    with pytest.raises(ValueError):
        repeated_failure_rate(0, 0)
