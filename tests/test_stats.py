"""Test seam 3 (part): statistics against reference values from statsmodels (Wilson)."""

import pytest

from ip_hack_detectors.stats import wilson


@pytest.mark.parametrize(
    "k,n,lo,hi",
    [
        (5, 10, 0.236593, 0.763407),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (30, 100, 0.218949, 0.395849),
    ],
)
def test_wilson_known_values(k, n, lo, hi):
    got_lo, got_hi = wilson(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-5)
    assert got_hi == pytest.approx(hi, abs=1e-5)


def test_wilson_empty_sample():
    assert wilson(0, 0) == (0.0, 1.0)
