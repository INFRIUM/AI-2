"""Indicators, thresholds, scores and grades (prereg 3, 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

GRADES = ("정상", "주의", "경고", "위험")
SEED = 20261109
BOOTSTRAP_ROUNDS = 2000


def ar_columns(axes: str) -> list[str]:
    """AR(5) coefficient columns for both sensors, e.g. axes='z' or 'yz'."""
    return [
        f"acc0{sensor}_ar{order}_{axis}"
        for axis in axes
        for sensor in (1, 2)
        for order in range(1, 6)
    ]


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def mahalanobis(reference: pd.DataFrame, target: pd.DataFrame, columns: list[str]):
    """Distance of each target row from the reference block (pinv covariance)."""
    ref = _matrix(reference, columns)
    ref = ref[~np.isnan(ref).any(axis=1)]
    if len(ref) < 2:
        raise ValueError("reference block needs at least 2 complete rows")
    mean = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False, ddof=1)
    inverse = np.linalg.pinv(cov)
    x = _matrix(target, columns) - mean
    distance = np.sqrt(np.einsum("ij,jk,ik->i", x, inverse, x).clip(min=0))
    distance[np.isnan(x).any(axis=1)] = np.nan
    info = {"n_reference": int(len(ref)), "cov_condition": float(np.linalg.cond(cov))}
    return pd.Series(distance, index=target.index), info


def standardized_euclid(reference: pd.DataFrame, target: pd.DataFrame, columns: list[str]):
    ref = _matrix(reference, columns)
    ref = ref[~np.isnan(ref).any(axis=1)]
    if len(ref) < 2:
        raise ValueError("reference block needs at least 2 complete rows")
    mean = ref.mean(axis=0)
    std = ref.std(axis=0, ddof=1)
    if (std <= 0).any():
        raise ValueError("zero reference spread")
    z = (_matrix(target, columns) - mean) / std
    distance = np.sqrt((z ** 2).sum(axis=1))
    distance[np.isnan(z).any(axis=1)] = np.nan
    return pd.Series(distance, index=target.index), {"n_reference": int(len(ref))}


def _clean(values) -> np.ndarray:
    array = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), float)
    return array[~np.isnan(array)]


def lower_thresholds(calib_values):
    values = _clean(calib_values)
    if len(values) == 0:
        return None, None
    return float(np.median(values)), float(np.percentile(values, 95))


def t1_uncertainty(calib_values, rounds: int = BOOTSTRAP_ROUNDS, seed: int = SEED):
    values = _clean(calib_values)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.percentile(rng.choice(values, size=len(values), replace=True), 95)
        for _ in range(rounds)
    ])
    top = np.sort(values)[::-1]
    return {
        "n": int(len(values)),
        "max": float(top[0]),
        "top3": [float(v) for v in top[:3]],
        "p95_bootstrap_90ci": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
    }


def upper_thresholds(grade_values: dict):
    """T2, T3, T4 from pooled medians of 주의/경고/위험 windows."""
    medians = {}
    for grade in ("주의", "경고", "위험"):
        values = _clean(grade_values.get(grade, []))
        medians[grade] = float(np.median(values)) if len(values) else None
    if any(v is None for v in medians.values()):
        return None, None, None, medians
    t2 = (medians["주의"] + medians["경고"]) / 2
    t3 = (medians["경고"] + medians["위험"]) / 2
    t4 = medians["위험"]
    return t2, t3, t4, medians


def thresholds_valid(thresholds) -> bool:
    if thresholds is None or any(t is None for t in thresholds):
        return False
    t0, t1, t2, t3, t4 = thresholds
    return t1 < t2 < t3 < t4


def score(value: float, thresholds) -> float | None:
    if value is None or np.isnan(value) or not thresholds_valid(thresholds):
        return None
    points = np.array(thresholds, float)
    if not points[0] < points[1]:
        return None
    return float(np.clip(np.interp(value, points, [0, 25, 50, 75, 100]), 0, 100))


def grade(value: float, thresholds) -> str | None:
    if value is None or np.isnan(value) or not thresholds_valid(thresholds):
        return None
    _, t1, t2, t3, _ = thresholds
    if value <= t1:
        return "정상"
    if value <= t2:
        return "주의"
    if value <= t3:
        return "경고"
    return "위험"


def in_band(value: float, expected: str, thresholds) -> bool:
    return grade(value, thresholds) == expected


def median(values) -> float | None:
    array = _clean(values)
    return float(np.median(array)) if len(array) else None


def exceed_rate(values, t1):
    array = _clean(values)
    if t1 is None or len(array) == 0:
        return {"n": int(len(array)), "exceed": None, "rate": None}
    exceed = int((array > t1).sum())
    return {"n": int(len(array)), "exceed": exceed, "rate": exceed / len(array)}
