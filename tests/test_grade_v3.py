"""Synthetic tests for grade v3 (plan A) rule functions."""

import numpy as np
import pandas as pd
import pytest

from src.public_grade import grade_v3 as gv


def test_bounds_p95_and_max():
    b = gv.bounds(np.arange(1, 101))
    assert b.low == pytest.approx(np.percentile(np.arange(1, 101), 95))
    assert b.high == 100 and b.n == 100


def test_bounds_rejects_empty_reference():
    with pytest.raises(ValueError):
        gv.bounds([np.nan])


def test_level_three_steps_and_missing():
    b = gv.Bounds(low=1.0, high=2.0, n=10)
    lv = gv.level(pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, np.nan]), b)
    assert lv.iloc[:5].tolist() == [0, 0, 1, 1, 2]
    assert np.isnan(lv.iloc[5])


def test_sustained_second_largest_of_three_consecutive():
    lv = pd.Series([2, 0, 2, 1, 0, 0], dtype=float)
    t = pd.Series(np.arange(6) * 10.0)
    s = gv.sustained(lv, t, pd.Series(["a"] * 6), step=10.0)
    assert np.isnan(s.iloc[0]) and np.isnan(s.iloc[1])
    assert s.iloc[2:].tolist() == [2, 1, 1, 0]


def test_sustained_breaks_on_gap_and_group():
    lv = pd.Series([2, 2, 2, 2, 2, 2], dtype=float)
    t = pd.Series([0, 10, 20, 40, 50, 60], dtype=float)          # gap between 20 and 40
    s = gv.sustained(lv, t, pd.Series(["a"] * 6), step=10.0)
    assert s.iloc[2] == 2 and np.isnan(s.iloc[3]) and np.isnan(s.iloc[4]) and s.iloc[5] == 2
    s2 = gv.sustained(lv.iloc[:3], t.iloc[:3], pd.Series(["a", "b", "b"]), step=10.0)
    assert s2.isna().all()


def test_grade_sequential_mapping():
    e = pd.Series([0, 1, 1, 2, 0, np.nan], dtype=float)
    s = pd.Series([np.nan, 0, 1, 2, 1, np.nan], dtype=float)
    assert gv.grade_sequential(e, s).tolist() == ["정상", "주의", "경고", "위험", "경고", gv.UNDECIDED]


def test_grade_single_agreement_rule():
    levels = pd.DataFrame({"a": [0, 1, 1, 2, 2, np.nan], "b": [0, 0, 1, 0, 2, np.nan]}, dtype=float)
    assert gv.grade_single(levels).tolist() == ["정상", "주의", "경고", "경고", "위험", gv.UNDECIDED]


def test_evidence_lists_exceeding_indicators():
    levels = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, np.nan, 2.0]})
    ev = gv.evidence(levels, {"x": "X", "y": "Y"})
    assert ev.tolist() == ["Y p95 초과", "X p95 초과", "X 정상 최댓값 초과, Y 정상 최댓값 초과"]
