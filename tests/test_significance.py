import numpy as np

from src.significance import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_brackets_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.05, size=200).clip(0, 1).tolist()
    ci = bootstrap_ci(values, n_resamples=500, ci=0.95, seed=1)
    assert ci.lo <= ci.mean <= ci.hi
    assert ci.lo < 0.5 < ci.hi
    # CI width should be small with n=200 and low variance.
    assert (ci.hi - ci.lo) < 0.05


def test_bootstrap_ci_empty_returns_zero():
    ci = bootstrap_ci([])
    assert ci.mean == 0.0 and ci.lo == 0.0 and ci.hi == 0.0


def test_paired_bootstrap_detects_real_improvement():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 0.4, size=30).tolist()
    b = (np.asarray(a) + 0.25).clip(0, 1).tolist()  # uniformly +0.25 better
    result = paired_bootstrap_test("A", "B", "metric", a, b, n_resamples=2000, seed=1)
    assert result.mean_diff > 0.2
    assert result.p_value < 0.05
    assert result.b_beats_a


def test_paired_bootstrap_null_when_no_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.3, 0.5, size=30).tolist()
    b = list(a)  # identical
    result = paired_bootstrap_test("A", "B", "metric", a, b, n_resamples=1000, seed=1)
    assert result.mean_diff == 0.0
    assert not result.b_beats_a


def test_paired_bootstrap_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap_test("A", "B", "m", [0.1, 0.2], [0.1], n_resamples=10)
