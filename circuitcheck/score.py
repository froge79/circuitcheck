"""Normalized faithfulness F = (m_C - m_empty) / (m_full - m_empty), with a denominator policy.

Every logit difference m is a difference of two logits, each rounded to the model dtype.
``res`` is the dtype's spacing at the largest |logit| involved, so each m carries a
rounding error of at most ``res``. To first order that bounds the error of F by
``2 * res * (1 + |F|) / |d|``. The score is undefined when d is non-finite or spans
fewer than ``k_min`` representable steps; scores are never clipped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

DEFAULT_K_MIN = 4.0
DEFAULT_F_TOL = 0.05
DEFAULT_WEAK_REL = 0.1


def resolution(eps: float, magnitude: float) -> float:
    """Spacing of a float format with machine epsilon ``eps`` at ``magnitude``."""
    if not math.isfinite(magnitude):
        return math.nan
    if magnitude <= 0:
        return eps
    return 2.0 ** math.floor(math.log2(magnitude)) * eps


@dataclass
class ScoreResult:
    m_full: float
    m_c: float
    m_empty: float
    d: float
    resolution: float
    f: float | None
    err_bound: float | None
    valid: bool
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def faithfulness(
    m_full: float,
    m_c: float,
    m_empty: float,
    res: float,
    k_min: float = DEFAULT_K_MIN,
    f_tol: float = DEFAULT_F_TOL,
    weak_rel: float = DEFAULT_WEAK_REL,
) -> ScoreResult:
    d = m_full - m_empty
    out = ScoreResult(m_full, m_c, m_empty, d, res, None, None, False)

    values = {"m_full": m_full, "m_C": m_c, "m_empty": m_empty, "resolution": res}
    bad = [k for k, v in values.items() if not math.isfinite(v)]
    if bad:
        out.reason = f"non-finite input: {', '.join(bad)}"
        return out
    if d == 0:
        out.reason = "denominator m_full - m_empty is exactly zero"
        return out
    if abs(d) < k_min * res:
        out.reason = (
            f"|d| = {abs(d):.6g} is below k_min * resolution = {k_min} * {res:.6g}; "
            "the denominator is within rounding of zero"
        )
        return out

    f = (m_c - m_empty) / d
    out.f = f
    out.valid = True
    out.err_bound = 2 * res * (1 + abs(f)) / abs(d)
    if out.err_bound > f_tol:
        out.warnings.append("LOW_PRECISION")
    if abs(d) < weak_rel * abs(m_full):
        out.warnings.append("WEAK_BASELINE")
    if d < 0:
        out.warnings.append("NEGATIVE_DENOMINATOR")
    if f < 0 or f > 1:
        out.warnings.append("OUT_OF_RANGE")
    return out
