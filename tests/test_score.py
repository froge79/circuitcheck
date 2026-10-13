import math

import pytest

from circuitcheck.score import faithfulness, resolution

BF16_EPS = 2.0**-7
RES = resolution(BF16_EPS, 20.0)  # 0.125 for logits in [16, 32)


def test_resolution_matches_bf16_steps_seen_in_spikes():
    assert RES == 0.125
    assert resolution(BF16_EPS, 9.0) == 0.0625
    assert resolution(2.0**-23, 20.0) == 16 * 2.0**-23


def test_known_denominator_spike_numbers():
    s = faithfulness(9.0, 7.71875, 3.8125, RES)
    assert s.valid and s.reason is None
    assert s.d == 5.1875
    assert s.f == pytest.approx(3.90625 / 5.1875)
    assert s.f == pytest.approx(0.7530, abs=1e-4)
    assert s.err_bound == pytest.approx(2 * 0.125 * (1 + s.f) / 5.1875)
    assert "LOW_PRECISION" in s.warnings  # err ~0.084 > 0.05, matching the bf16/fp32 gap


def test_invariants_f_is_one_and_zero():
    assert faithfulness(9.0, 9.0, 3.8125, RES).f == 1.0
    assert faithfulness(9.0, 3.8125, 3.8125, RES).f == 0.0


def test_zero_denominator_is_undefined():
    s = faithfulness(5.0, 4.0, 5.0, RES)
    assert not s.valid and s.f is None and s.d == 0
    assert "exactly zero" in s.reason


@pytest.mark.parametrize("factor,valid", [(0.9, False), (1.1, True)])
def test_near_zero_denominator_threshold(factor, valid):
    k_min = 4.0
    d = k_min * RES * factor
    s = faithfulness(3.0 + d, 3.0 + d / 2, 3.0, RES, k_min=k_min)
    assert s.valid is valid
    assert s.d == pytest.approx(d)
    if not valid:
        assert s.f is None and "k_min" in s.reason
    else:
        assert s.f == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_inputs_are_undefined(bad):
    for args in [(bad, 1.0, 0.0), (5.0, bad, 0.0), (5.0, 1.0, bad)]:
        s = faithfulness(*args, RES)
        assert not s.valid and s.f is None and "non-finite" in s.reason
    s = faithfulness(5.0, 1.0, 0.0, bad)
    assert not s.valid and "resolution" in s.reason


def test_scores_outside_unit_interval_are_preserved():
    below = faithfulness(9.0, 1.0, 3.0, RES)  # (1-3)/6
    above = faithfulness(9.0, 12.0, 3.0, RES)  # 9/6
    assert below.f == pytest.approx(-1 / 3) and below.valid
    assert above.f == pytest.approx(1.5) and above.valid
    assert "OUT_OF_RANGE" in below.warnings and "OUT_OF_RANGE" in above.warnings


def test_negative_denominator_kept_with_warning():
    s = faithfulness(2.0, 4.0, 6.0, RES)  # d = -4, F = (4-6)/-4 = 0.5
    assert s.valid and s.f == pytest.approx(0.5)
    assert "NEGATIVE_DENOMINATOR" in s.warnings


def test_weak_baseline_warning():
    s = faithfulness(9.0, 8.8, 8.5, resolution(2.0**-23, 20.0))  # precise but tiny d
    assert s.valid and "WEAK_BASELINE" in s.warnings and "LOW_PRECISION" not in s.warnings
