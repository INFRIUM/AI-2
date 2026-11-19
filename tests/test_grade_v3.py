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


def test_recent_counts_last_three_windows():
    lv = pd.Series([2, 0, 1, 2, 2], dtype=float)
    t = pd.Series(np.arange(5) * 10.0)
    c = gv.recent_counts(lv, t, pd.Series(["a"] * 5), step=10.0)
    assert c["n_low"].isna().iloc[:2].all()
    assert c["n_low"].iloc[2:].tolist() == [2, 2, 3]
    assert c["n_high"].iloc[2:].tolist() == [1, 1, 2]


B = {"x": gv.Bounds(low=4.49, high=6.68, n=64), "y": gv.Bounds(low=0.7145, high=0.7308, n=60)}
N = {"x": "AR z 거리", "y": "1-TAC z"}


def test_reason_danger_shows_values_bounds_and_persistence():
    text = gv.reason("위험", {"x": 6.94, "y": 0.747}, {"x": 2.0, "y": 2.0}, N, B,
                     {"n_low": 3.0, "n_high": 2.0}, sequential=True)
    assert "AR z 거리 6.94: 위험 경계 6.68 초과" in text
    assert "1-TAC z 0.747: 위험 경계 0.731 초과" in text
    assert "최근 3개 Window 중 2개가 위험 경계 초과" in text


def test_reason_normal_lists_all_indicators_with_upper_bound():
    text = gv.reason("정상", {"x": 3.56, "y": 0.654}, {"x": 0.0, "y": 0.0}, N, B,
                     {"n_low": 0.0, "n_high": 0.0}, sequential=True)
    assert text == "모든 지표 정상 범위: AR z 거리 3.56 (정상 상한 4.49 이하), 1-TAC z 0.654 (정상 상한 0.715 이하)"


def test_reason_carried_grade_explains_current_window_is_normal():
    text = gv.reason("위험", {"x": 3.73, "y": 0.639}, {"x": 0.0, "y": 0.0}, N, B,
                     {"n_low": 2.0, "n_high": 2.0}, sequential=True)
    assert text.startswith("이번 Window는 모든 지표 정상 범위")
    assert "2개가 위험 경계 초과" in text


def test_reason_caution_without_history_and_held():
    text = gv.reason("주의", {"x": 5.0, "y": np.nan}, {"x": 1.0, "y": np.nan}, N, B,
                     {"n_low": np.nan, "n_high": np.nan}, sequential=True)
    assert "AR z 거리 5: 정상 상한 4.49 초과 (위험 경계 6.68 이하)" in text
    assert "지속 여부를 판단할 수 없음" in text
    assert gv.reason(gv.HELD, {}, {}, N, B, None, True).startswith("판정 보류")


def test_reason_single_window_agreement():
    text = gv.reason("경고", {"x": 7.0, "y": 0.70}, {"x": 2.0, "y": 0.0}, N, B, None, sequential=False)
    assert "정상 상한 초과 1개, 위험 경계 초과 1개" in text
