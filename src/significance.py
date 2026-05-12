"""Bootstrap confidence intervals and paired significance tests.

Used to attach error bars to retrieval metrics (Recall, MRR, nDCG) and to
ask "is strategy X significantly better than strategy Y on this labelled
set?" — a question raw point estimates can't answer with small n.

Non-parametric bootstrap is the right tool here: the per-query metrics
are bounded in [0, 1] and very non-normal, so a t-test would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    ci: float  # confidence level (e.g. 0.95)

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return BootstrapCI(mean=0.0, lo=0.0, hi=0.0, ci=ci)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return BootstrapCI(
        mean=float(arr.mean()),
        lo=float(np.quantile(boot_means, alpha)),
        hi=float(np.quantile(boot_means, 1 - alpha)),
        ci=ci,
    )


@dataclass(frozen=True)
class PairedTestResult:
    strategy_a: str
    strategy_b: str
    metric: str
    mean_diff: float  # b - a, positive ⇒ B wins
    p_value: float  # one-sided: H1 is "B > A"
    ci95: BootstrapCI  # CI of the per-query difference

    @property
    def b_beats_a(self) -> bool:
        return self.p_value < 0.05 and self.mean_diff > 0

    @property
    def a_beats_b(self) -> bool:
        return self.p_value > 0.95 and self.mean_diff < 0


def paired_bootstrap_test(
    strategy_a: str,
    strategy_b: str,
    metric: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> PairedTestResult:
    """Paired one-sided bootstrap test: is `B` better than `A`?

    Returns the mean per-query difference (B - A), a 95% CI on that
    difference, and a one-sided p-value for H1: mean_diff > 0.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have the same shape")

    diffs = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    # One-sided p-value for H1: mean_diff > 0
    p_value = float(np.mean(boot_means <= 0))
    ci = bootstrap_ci(diffs.tolist(), n_resamples=n_resamples, ci=0.95, seed=seed + 1)
    return PairedTestResult(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        metric=metric,
        mean_diff=float(diffs.mean()),
        p_value=p_value,
        ci95=ci,
    )
