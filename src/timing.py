"""Per-stage latency capture.

Use as a context manager:

    timing = Timing()
    with timing.stage("embed"):
        ...

Or accumulate manually with `timing.record(name, ms)`. Designed to be cheap
enough to leave on by default.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass
class Timing:
    stages_ms: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + (
                (time.perf_counter() - start) * 1000.0
            )

    def record(self, name: str, ms: float) -> None:
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return float(sum(self.stages_ms.values()))

    def to_dict(self) -> Dict[str, float]:
        out = {k: round(v, 3) for k, v in self.stages_ms.items()}
        out["total"] = round(self.total_ms, 3)
        return out


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dep)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac
