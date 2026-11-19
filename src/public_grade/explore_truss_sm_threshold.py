"""Truss v2 step 5-4: threshold for a ONE-DAY baseline on SM data (DEVELOPMENT ONLY).

Problem (5-2, 5-3): the SM P2 Z check uses _01 as the only normal baseline and
_02 as the false-alarm check. Within one recording there are no proper
replicates (log sweep, up/down windows do not share a frequency band), so the
threshold cannot come from the baseline day itself.

Candidate (plan C): learn the size of the day-to-day difference on the
development conditions (SM P1 Y, P1 Z, P2 Y; two days each) and transfer the
threshold to a condition that was not used to set it.

Indicator (time-aligned):
  x_D(t)   deg AR(5) z coefficients (10 dims) of day D, averaged over k
           consecutive usable windows ending at window start time t
  d(t)     sqrt(mean_j ((x_B(t)_j - x_A(t)_j) / s_j)^2)   (A = baseline day)
  scale s  base_std : std of day A's usable averaged windows (own condition)
           diff_std : std of day differences pooled over the conditions that
                      set the threshold (not the evaluated one)
  flag     d(t) > T, then m of n consecutive aligned windows

Transfer test (fixed before the first run, 2026-11-19): leave one condition
out. T = p95 / p99 / max of d over the other two conditions (both directions);
false-alarm rate on the left-out condition, both directions.
Grid: k 1, 2, 3 ; scale base_std, diff_std ; T p95, p99, max ;
persist 1/1, 2/3 ; data deg (main), raw (reference).
Also reported: T from all three conditions (the value a prereg would carry to
SM P2 Z). Damage detection cannot be assessed here (no damage data by design).

Usage: python -m src.public_grade.explore_truss_sm_threshold --data-dir data/raw/ai1_public_1.2_dev --out <dir>
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
from src.public_grade.explore_truss_sm import averaged, prepare

K_VALUES = (1, 2, 3)
SCALES = ("base_std", "diff_std")
QUANTILES = {"p95": 95, "p99": 99, "max": 100}
PERSIST = ((1, 1), (2, 3))
COLUMNS = g.ar_columns("z")


def aligned(frame: pd.DataFrame, data: pd.DataFrame, k: int) -> dict:
    """Per (condition, baseline day): aligned feature arrays of A and B."""
    values = data.loc[frame.index, COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = {}
    for cond, part in frame.groupby("condition"):
        days = {}
        for day, p in part.groupby("day"):
            avg = averaged(p, values, k)
            avg.index = p.loc[avg.index, "window_start_s"].round(2).to_numpy()
            days[day] = avg.sort_index()
        for base, other in (("01", "02"), ("02", "01")):
            a, b = days[base], days[other]
            common = a.index.intersection(b.index)
            both = common[a.loc[common].notna().all(axis=1) & b.loc[common].notna().all(axis=1)]
            out[(cond, base)] = {
                "t": np.asarray(both, float),
                "A": a.loc[both].to_numpy(float), "B": b.loc[both].to_numpy(float),
                "A_all": a.dropna().to_numpy(float),
            }
    return out


def distances(pair: dict, scale: np.ndarray) -> np.ndarray:
    diff = (pair["B"] - pair["A"]) / scale
    return np.sqrt(np.mean(diff ** 2, axis=1))


def scale_for(pair: dict, scale_kind: str, train_pairs: list[dict]) -> np.ndarray:
    if scale_kind == "base_std":
        s = pair["A_all"].std(axis=0, ddof=1)
    else:
        s = np.concatenate([p["B"] - p["A"] for p in train_pairs]).std(axis=0, ddof=1)
    s[s == 0] = 1.0
    return s


def persist(flags: np.ndarray, t: np.ndarray, m: int, n: int) -> np.ndarray:
    if n == 1:
        return flags.astype(float)
    out = np.full(len(flags), np.nan)
    step = 10.24
    for i in range(n - 1, len(flags)):
        span = t[i - n + 1:i + 1]
        if np.all(np.abs(np.diff(span) - step) < 1e-3):
            out[i] = float(flags[i - n + 1:i + 1].sum() >= m)
    return out


def explore(frame: pd.DataFrame, data: pd.DataFrame, kind: str) -> tuple[list, list]:
    rows, final = [], []
    conditions = sorted(frame["condition"].unique())
    for k in K_VALUES:
        pairs = aligned(frame, data, k)
        for scale_kind, (qname, q), (m, n) in itertools.product(SCALES, QUANTILES.items(), PERSIST):
            # leave one condition out
            for held in conditions:
                train = [pairs[(c, b)] for c in conditions if c != held for b in ("01", "02")]
                pooled = np.concatenate([distances(p, scale_for(p, scale_kind, train)) for p in train])
                T = float(np.percentile(pooled, q))
                for base in ("01", "02"):
                    p = pairs[(held, base)]
                    d = distances(p, scale_for(p, scale_kind, train))
                    flags = persist(d > T, p["t"], m, n)
                    valid = flags[~np.isnan(flags)]
                    rows.append({
                        "kind": kind, "k": k, "scale": scale_kind, "threshold": qname,
                        "persist": f"{m}/{n}", "held_out": held, "baseline_day": base,
                        "T": T, "n": int(len(valid)),
                        "fa": float(valid.mean()) if len(valid) else None,
                        "d_median": float(np.median(d)), "d_p95": float(np.percentile(d, 95)),
                    })
            # all three conditions: the value a pre-registration would carry over
            allp = [pairs[(c, b)] for c in conditions for b in ("01", "02")]
            pooled = np.concatenate([distances(p, scale_for(p, scale_kind, allp)) for p in allp])
            final.append({"kind": kind, "k": k, "scale": scale_kind, "threshold": qname,
                          "persist": f"{m}/{n}", "T_all": float(np.percentile(pooled, q)),
                          "n_pooled": int(len(pooled))})
    return rows, final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    raw, deg = loader.load_pair(args.data_dir, "hbta")
    frame = prepare(raw, deg)
    frame = frame[~frame["held"]]
    rows, final = [], []
    for kind, data in (("deg", deg), ("raw", raw)):
        r, f = explore(frame, data, kind)
        rows += r
        final += f
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "loco.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final).to_csv(args.out / "threshold_all.csv", index=False, encoding="utf-8-sig")
    (args.out / "meta.json").write_text(json.dumps({
        "note": "개발용 탐색 (결과 열람 후). 성능 근거로 쓰지 않음",
        "files": {p.name: loader.file_digest(p) for p in sorted(Path(args.data_dir).glob("hbta_*.csv"))},
    }, ensure_ascii=False, indent=1))
    print(f"{len(rows)} LOCO rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
