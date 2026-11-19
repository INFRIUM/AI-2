"""Synthetic tests for the Truss v2 evaluation (no public data needed)."""

import numpy as np
import pandas as pd
import pytest

from src.public_grade import truss_v2 as tv

STEP = tv.WINDOW_S


def frame(values: np.ndarray, held=None, start: float = 0.0) -> pd.DataFrame:
    n = len(values)
    data = {col: values[:, i] for i, col in enumerate(tv.AR_Z)}
    out = pd.DataFrame(data)
    out["held"] = np.zeros(n, bool) if held is None else np.asarray(held, bool)
    out.index = pd.Index(np.round(start + np.arange(n) * STEP, 2), name="t")
    return out


def test_distance_zero_for_identical_records():
    rng = np.random.default_rng(0)
    base = frame(rng.normal(size=(20, 10)))
    d = tv.distance(base, base.copy(), tv.AR_Z)
    assert len(d) == 20
    assert np.allclose(d, 0.0)


def test_distance_known_shift_in_baseline_std_units():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(30, 10))
    base = frame(values)
    s = values.std(axis=0, ddof=1)
    target = frame(values + 2 * s)          # every coefficient moves by 2 baseline std
    d = tv.distance(base, target, tv.AR_Z)
    assert np.allclose(d, 2.0)


def test_distance_uses_only_windows_usable_in_both():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(10, 10))
    held_b = [False] * 10
    held_b[3] = True
    held_t = [False] * 10
    held_t[7] = True
    d = tv.distance(frame(values, held_b), frame(values + 1, held_t), tv.AR_Z)
    assert set(d.index) == set(frame(values).index) - {round(3 * STEP, 2), round(7 * STEP, 2)}


def test_distance_scale_ignores_held_baseline_windows():
    values = np.ones((6, 10))
    values[:, 0] = [0, 1, 2, 3, 100, 4]           # held outlier must not inflate the std
    held = [False, False, False, False, True, False]
    s = tv.scale(frame(values, held), tv.AR_Z)
    assert s[0] == pytest.approx(np.std([0, 1, 2, 3, 4], ddof=1))
    assert np.all(s[1:] == 1.0)                    # zero spread -> 1


def test_shift_aligns_target_by_whole_windows():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(15, 10))
    base = frame(values)
    late = frame(np.vstack([rng.normal(size=(1, 10)), values]))   # target one window late
    assert tv.distance(base, late, tv.AR_Z, shift_windows=1).max() == pytest.approx(0.0)
    assert tv.distance(base, late, tv.AR_Z).max() > 0.1


def test_persist_two_of_three_and_undecided():
    t = np.round(np.arange(6) * STEP, 2)
    d = pd.Series([3, 0, 3, 0, 0, 3], index=t, dtype=float)
    flags = tv.persist(d, threshold=1.0)
    assert np.isnan(flags.iloc[0]) and np.isnan(flags.iloc[1])
    assert flags.iloc[2:].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_persist_requires_consecutive_windows():
    t = np.round(np.array([0, 1, 2, 4, 5, 6]) * STEP, 2)        # gap after the third window
    d = pd.Series([3, 3, 3, 3, 3, 3], index=t, dtype=float)
    flags = tv.persist(d, threshold=1.0)
    assert np.isnan(flags.iloc[3]) and np.isnan(flags.iloc[4])
    assert flags.iloc[2] == 1.0 and flags.iloc[5] == 1.0


def test_rate_excludes_undecided():
    r = tv.rate(pd.Series([np.nan, 1.0, 0.0, 0.0]))
    assert r == {"n": 3, "anomaly": 1, "rate": pytest.approx(1 / 3), "undecided": 1}


def test_auc_perfect_and_ties():
    assert tv.auc([0, 1], [2, 3]) == 1.0
    assert tv.auc([1, 1], [1, 1]) == 0.5
    assert tv.auc([], [1]) is None


def _shaker(freqs, rms=10.0):
    t = np.round(np.arange(len(freqs)) * STEP, 2)
    return pd.DataFrame({"as_dom_freq_hz": freqs, "as_rms_m_s2": rms}, index=pd.Index(t, name="t"))


def test_best_lag_detects_one_window_delay():
    sweep = 2.0 * 1.12 ** np.arange(30)
    base = _shaker(sweep)
    late = _shaker(np.concatenate([[2.0], sweep[:-1]]))
    assert tv.best_lag(base, base)["best_lag"] == 0
    out = tv.best_lag(base, late)
    assert out["best_lag"] == 1 and out["misaligned"]


def test_best_lag_ignores_windows_with_shaker_off():
    sweep = 2.0 * 1.12 ** np.arange(30)
    base = _shaker(sweep)
    noisy = sweep.copy()
    noisy[:5] = [40, 3, 30, 5, 20]                  # junk while shaker is off
    target = _shaker(noisy)
    target.iloc[:5, target.columns.get_loc("as_rms_m_s2")] = 0.1
    assert tv.best_lag(base, target)["best_lag"] == 0


def test_check_thresholds_stops_on_mismatch(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tv, "dev_pooled", lambda dev, data, cols: np.array([1.0, 2.0, 3.0]))
    with pytest.raises(tv.CheckError):
        tv.check_thresholds(SimpleNamespace(deg=None, raw=None))


def test_check_thresholds_accepts_preregistered_values(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tv, "dev_pooled", lambda dev, data, cols: np.array(
        [tv.T_DEG] * 5 if data == "deg" else [tv.T_RAW] * 5))
    out = tv.check_thresholds(SimpleNamespace(deg="deg", raw="raw"))
    assert out["deg"]["recomputed"] == pytest.approx(tv.T_DEG)
