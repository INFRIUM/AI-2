"""Truss v2 exploration on AI1-public-1.1 HBTA NM data (DEVELOPMENT ONLY).

All 10 NM recordings were already evaluated in v1, so every number produced
here is "결과 열람 후 탐색" and must never be cited as performance evidence.
Its only purpose is to choose candidates for the Truss v2 pre-registration,
which will be evaluated once on unseen SM recordings.

Exploration grid (fixed before the first run, 2026-11-19; every combination
is reported, not only the best):

  features   ARz   : deg AR(5) z of acc01, acc02 (10 dims)       [distance]
             ARyz  : deg AR(5) y+z (20 dims), only if y present  [distance]
             TACz  : 1-TAC z (tac_a_b_z), value itself            [scalar]
  k          consecutive 10.24 s windows averaged: 1, 2, 3, 4, 6, 8
  cov        sample (pinv), ledoit_wolf (on standardized features), diag
  threshold  p95 or max of the calibration distances
  persist    m of n consecutive decisions: 1/1, 2/3, 3/5
  scheme     v1   : fit AI① train (both UDS), calib ai2_calib, FA = UDS test
             x01  : fit UDS_01 first 49 windows, calib UDS_01 last 33,
                    FA = all UDS_02 windows   (mirrors the SM plan)
             x02  : same with UDS_01 and UDS_02 swapped
  data       deg (main) and raw (ceiling, reference only)

Pass (binary normal/anomaly, v1 conditions 1 and 2):
  FA rate <= 5 % on the FA set, detection >= 95 % on danger states DS2, DS8.
Detection over all DS1-DS8 is reported as well.

Aggregation never crosses a split/role boundary or a time gap. Windows that
cannot form a full k-block or a full n-sequence are dropped (causal: the
first decisions of a run are not available yet).

Usage: python -m src.public_grade.explore_truss_nm --data-dir data/raw/ai1_public_1.1 --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import grading as g
from src.public_grade import loader

WINDOW_S = 10.24
K_VALUES = (1, 2, 3, 4, 6, 8)
COVS = ("sample", "ledoit_wolf", "diag")
THRESHOLDS = ("p95", "max")
PERSIST = ((1, 1), (2, 3), (3, 5))
SCHEMES = ("v1", "x01", "x02")
DANGER = ("DS2", "DS8")
FA_LIMIT, DETECT_MIN = 0.05, 0.95
FIT_WINDOWS = 49   # x-schemes: first 49 windows of the reference UDS recording (= AI① train + val time)


# --------------------------------------------------------------- roles, runs

def assign_roles(frame: pd.DataFrame, scheme: str) -> pd.Series:
    role = pd.Series(None, index=frame.index, dtype=object)
    uds = frame["damage_stage"] == "UDS"
    role[~uds] = "det"
    if scheme == "v1":
        role[uds & (frame["ai1_split"] == "train")] = "fit"
        role[uds & (frame["ai1_split"] == "ai2_calib")] = "calib"
        role[uds & (frame["ai1_split"] == "test")] = "fa"
        return role
    ref, other = ("_01", "_02") if scheme == "x01" else ("_02", "_01")
    rec = frame["source_record"].astype(str)
    for name in rec[uds & rec.str.endswith(ref)].unique():
        idx = frame.index[rec == name]
        order = frame.loc[idx, "window_start_us"].sort_values().index
        role[order[:FIT_WINDOWS]] = "fit"
        role[order[FIT_WINDOWS:]] = "calib"
    role[uds & rec.str.endswith(other)] = "fa"
    return role


def add_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Run id: same recording, same role, consecutive 10.24 s windows."""
    frame = frame[frame["role"].notna()].sort_values(["source_record", "window_start_us"]).copy()
    run, last = [], {}
    counter = 0
    for rec, role, start in zip(frame["source_record"], frame["role"], frame["window_start_us"]):
        prev = last.get(rec)
        if prev is None or prev[0] != role or abs((start - prev[1]) / 1e6 - WINDOW_S) > 1e-3:
            counter += 1
        run.append(counter)
        last[rec] = (role, start)
    frame["run"] = run
    return frame


def aggregate(frame: pd.DataFrame, columns: list[str], k: int) -> pd.DataFrame:
    if k == 1:
        out = frame.copy()
    else:
        parts = []
        for _, part in frame.groupby("run", sort=False):
            block = part.copy()
            block[columns] = part[columns].rolling(k, min_periods=k).mean().to_numpy()
            parts.append(block.iloc[k - 1:])
        out = pd.concat(parts) if parts else frame.iloc[0:0]
    return out.dropna(subset=columns)


# --------------------------------------------------------------- distances

def _ledoit_wolf(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / n
    m = np.trace(s) / p
    d2 = np.sum((s - m * np.eye(p)) ** 2) / p
    b_bar2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc) / (n ** 2) / p
    shrink = 0.0 if d2 == 0 else min(b_bar2, d2) / d2
    return shrink * m * np.eye(p) + (1 - shrink) * s


def distance(fit: pd.DataFrame, target: pd.DataFrame, columns: list[str], cov: str) -> np.ndarray:
    ref = fit[columns].to_numpy(float)
    x = target[columns].to_numpy(float)
    if len(columns) == 1:            # scalar indicator (1-TAC): the value itself
        return x[:, 0]
    mean, std = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    zr, zx = (ref - mean) / std, (x - mean) / std
    if cov == "diag":
        return np.sqrt((zx ** 2).sum(axis=1))
    sigma = np.cov(zr, rowvar=False, ddof=1) if cov == "sample" else _ledoit_wolf(zr)
    inv = np.linalg.pinv(sigma)
    return np.sqrt(np.einsum("ij,jk,ik->i", zx, inv, zx).clip(min=0))


def persist(flags: pd.Series, runs: pd.Series, m: int, n: int) -> pd.Series:
    if n == 1:
        return flags.astype(float)
    out = pd.Series(np.nan, index=flags.index)
    for _, idx in flags.groupby(runs, sort=False).groups.items():
        f = flags.loc[idx].astype(float)
        total = f.rolling(n, min_periods=n).sum()
        out.loc[idx] = np.where(total.isna(), np.nan, (total >= m).astype(float))
    return out


# --------------------------------------------------------------- one combo

def rate(values: pd.Series) -> tuple[float | None, int]:
    v = values.dropna()
    return (float(v.mean()) if len(v) else None), int(len(v))


def run_combo(frame: pd.DataFrame, columns, k, cov, thr, m, n):
    agg = aggregate(frame, columns, k)
    fit = agg[agg["role"] == "fit"]
    if len(fit) < max(3, len(columns) + 1) and len(columns) > 1:
        cov_used = "diag" if cov == "sample" else cov
    else:
        cov_used = cov
    d = pd.Series(distance(fit, agg, columns, cov_used), index=agg.index)
    calib = d[agg["role"] == "calib"].dropna()
    if calib.empty:
        return None
    t1 = float(np.percentile(calib, 95)) if thr == "p95" else float(calib.max())
    flags = persist(d > t1, agg["run"], m, n)
    fa, n_fa = rate(flags[agg["role"] == "fa"])
    det = agg["role"] == "det"
    danger, n_danger = rate(flags[det & agg["damage_stage"].isin(DANGER)])
    alld, n_all = rate(flags[det])
    by_stage = {s: rate(flags[det & (agg["damage_stage"] == s)])[0] for s in sorted(agg.loc[det, "damage_stage"].unique())}
    return {
        "n_fit": int(len(fit)), "n_calib": int(len(calib)), "cov_used": cov_used, "T1": t1,
        "fa": fa, "n_fa": n_fa, "danger": danger, "n_danger": n_danger,
        "all_damage": alld, "n_all_damage": n_all, "by_stage": by_stage,
        "span_s": round((k + n - 1) * WINDOW_S, 2),
        "pass": fa is not None and danger is not None and fa <= FA_LIMIT and danger >= DETECT_MIN,
    }


def feature_sets(frame: pd.DataFrame) -> dict:
    sets = {"ARz": g.ar_columns("z"), "TACz": ["tac_a_b_z"]}
    yz = g.ar_columns("yz")
    if frame[yz].notna().all(axis=1).mean() > 0.9:
        sets["ARyz"] = yz
    return sets


def explore(data_dir: Path) -> pd.DataFrame:
    raw, deg = loader.load_pair(data_dir, "hbta")
    rows = []
    for kind, base in (("deg", deg), ("raw", raw)):
        base = base[~base["held_main"]].copy()
        for scheme in SCHEMES:
            frame = base.copy()
            frame["role"] = assign_roles(frame, scheme).to_numpy()
            frame = add_runs(frame)
            for fname, cols in feature_sets(base).items():
                for k, cov, thr, (m, n) in itertools.product(K_VALUES, COVS, THRESHOLDS, PERSIST):
                    if fname == "TACz" and cov != "sample":
                        continue      # scalar: covariance option irrelevant
                    res = run_combo(frame, cols, k, cov, thr, m, n)
                    if res is None:
                        continue
                    rows.append({"kind": kind, "scheme": scheme, "feature": fname, "k": k,
                                 "cov": cov if fname != "TACz" else "-", "threshold": thr,
                                 "persist": f"{m}/{n}", **res})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = explore(args.data_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    flat = table.drop(columns=["by_stage"]).copy()
    stages = pd.DataFrame(table["by_stage"].tolist()).add_prefix("det_")
    flat = pd.concat([flat.reset_index(drop=True), stages], axis=1)
    flat.to_csv(args.out / "explore_truss_nm.csv", index=False, encoding="utf-8-sig")
    meta = {"note": "개발용 탐색 (결과 열람 후). 성능 근거로 쓰지 않음",
            "data": str(args.data_dir), "n_combinations": int(len(flat)),
            "files": {p.name: loader.file_digest(p) for p in sorted(Path(args.data_dir).glob("hbta_*.csv"))}}
    (args.out / "explore_truss_nm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{len(flat)} combinations → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
