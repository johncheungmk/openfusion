from __future__ import annotations

import math


WILSON_95_Z = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    bounded_successes = max(0, min(successes, total))
    proportion = bounded_successes / total
    z_squared = WILSON_95_Z**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        WILSON_95_Z
        * math.sqrt(
            (proportion * (1 - proportion) + z_squared / (4 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
