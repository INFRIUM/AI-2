"""Truss v2 exploration on SM (sweep) normal recordings, AI1-public-1.2-dev (DEVELOPMENT ONLY).

Six normal SM recordings (P1 Y, P1 Z, P2 Y; two days each). No damage data.
Every number is "결과 열람 후 탐색" and is not performance evidence.

Questions (fixed before the first run, 2026-11-19):
  Q1 hold    : which time ranges of the sweep are held in the degraded data,
               and are they the same on both days?
  Q2 timing  : does the sweep run at the same time on both days? (dominant
               frequency of the excitation-axis response at the same window
               start time; AS data not yet available)
  Q3 day gap : is the day-to-day difference larger than the within-day
               difference? This is what decides whether a one-day baseline
               (SM P2 Z plan: _01 baseline, _02 false-alarm check) can work.

Q3 method
  pairs      within-day : up-sweep window vs down-sweep window of the SAME
                          recording at the same excitation frequency (nearest
                          raw dominant frequency, |df| <= 0.25 Hz, both usable)
             cross-day  : window at the same start time on the other day
                          (both usable)
  usable     not held in the degraded data on both members of a pair
  features   ARz (10 dims) ; ARyz (20 dims, pairs where y is present)
  k          1, 2, 3 consecutive usable windows averaged (same run only)
  distance   Euclidean on features standardized with the std of all usable
             windows of that condition
  rule       baseline day A only: T = p95 or max of A's within-day distances;
             cross-day false-alarm = share of A-B same-time distances > T
             (both directions A=_01/B=_02 and A=_02/B=_01)
  data       deg (main) and raw (reference)

Usage: python -m src.public_grade.explore_truss_sm --data-dir data/raw/ai1_public_1.2_dev --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import grading as g
from src.public_grade import loader

WINDOW_S = 10.24
DF_MAX = 0.25
K_VALUES = (1, 2, 3)


def excitation_axis(condition: str) -> str:
    return condition.rsplit("_", 1)[-1].lower()


def attach_as(frame: pd.DataFrame, as_path: Path) -> pd.DataFrame:
    """Use the shaker accelerometer (AS) dominant frequency for pairing (added
    2026-11-19 after AI① delivered hbta_as_dev.csv; bridge-based run kept)."""
    shaker = pd.read_csv(as_path, encoding="utf-8-sig")
    keys = ["experiment_id", "window_start_s"]
    merged = frame.merge(shaker[keys + ["as_dom_freq_hz"]], on=keys, how="left", validate="one_to_one")
    merged.index = frame.index
    merged["bridge_dom_f"] = merged["dom_f"]
    merged["dom_f"] = merged["as_dom_freq_hz"]
    return merged


def prepare(raw: pd.DataFrame, deg: pd.DataFrame) -> pd.DataFrame:
    """One row per window with day tag, usability and dominant frequency."""
    keep = ["experiment_id", "source_record", "condition", "window_start_s"]
    out = raw[keep].copy()
    out["day"] = out["source_record"].str[-2:]
    out["held"] = deg["held_main"].to_numpy()
    out["y_held"] = (deg["hold_acc01_y"].astype(str) != "False").to_numpy() | (
        deg["hold_acc02_y"].astype(str) != "False").to_numpy()
    out["dom_f"] = [
        raw.at[i, f"acc01_dom_freq_{excitation_axis(c)}"] for i, c in zip(raw.index, raw["condition"])
    ]
    out["rms"] = [raw.at[i, f"acc01_rms_{excitation_axis(c)}"] for i, c in zip(raw.index, raw["condition"])]
    return out


# ------------------------------------------------------------------ Q1, Q2

def usable_ranges(frame: pd.DataFrame) -> dict:
    out = {}
    for rec, part in frame.groupby("source_record"):
        part = part.sort_values("window_start_s")
        ok = ~part["held"].to_numpy()
        t = part["window_start_s"].to_numpy()
        ranges, start = [], None
        for i, flag in enumerate(ok):
            if flag and start is None:
                start = t[i]
            if (not flag or i == len(ok) - 1) and start is not None:
                end = t[i] + WINDOW_S if flag else t[i]
                ranges.append([round(float(start), 2), round(float(end), 2)])
                start = None
        out[rec] = {"n_windows": int(len(part)), "n_usable": int(ok.sum()), "usable_s": ranges}
    return out


def timing(frame: pd.DataFrame) -> dict:
    out = {}
    for cond, part in frame.groupby("condition"):
        a = part[part["day"] == "01"].set_index("window_start_s")
        b = part[part["day"] == "02"].set_index("window_start_s")
        common = a.index.intersection(b.index)
        both = common[(~a.loc[common, "held"]) & (~b.loc[common, "held"])]
        df = (a.loc[both, "dom_f"] - b.loc[both, "dom_f"]).abs()
        ra, rb = np.log(a.loc[common, "rms"]), np.log(b.loc[common, "rms"])
        lags = {}
        for lag in range(-3, 4):
            x, y = ra.to_numpy(), np.roll(rb.to_numpy(), lag)
            lags[lag] = float(np.corrcoef(x[3:-3], y[3:-3])[0, 1])
        out[cond] = {
            "n_both_usable": int(len(both)),
            "dom_f_abs_diff_median": float(df.median()), "dom_f_abs_diff_p90": float(df.quantile(0.9)),
            "share_within_0p5Hz": float((df <= 0.5).mean()),
            "log_rms_corr_by_lag_windows": lags,
            "best_lag_windows": int(max(lags, key=lags.get)),
        }
    return out


# ------------------------------------------------------------------ Q3

def runs(part: pd.DataFrame) -> pd.Series:
    t = part["window_start_s"].to_numpy()
    usable = ~part["held"].to_numpy()
    run_id, current, prev_t = [], 0, None
    for ti, u in zip(t, usable):
        if not u:
            run_id.append(-1)
            prev_t = None
            continue
        if prev_t is None or abs(ti - prev_t - WINDOW_S) > 1e-3:
            current += 1
        run_id.append(current)
        prev_t = ti
    return pd.Series(run_id, index=part.index)


def averaged(part: pd.DataFrame, values: pd.DataFrame, k: int) -> pd.DataFrame:
    """Average features over k consecutive usable windows; row = last window."""
    part = part.sort_values("window_start_s")
    r = runs(part)
    vals = values.loc[part.index]
    out = pd.DataFrame(np.nan, index=part.index, columns=values.columns)
    for run_id, idx in r.groupby(r).groups.items():
        if run_id < 0:
            continue
        block = vals.loc[idx].rolling(k, min_periods=k).mean()
        out.loc[idx] = block.to_numpy()
    return out


def pairs_within(part: pd.DataFrame) -> list[tuple]:
    part = part[~part["held"]].sort_values("window_start_s")
    if part.empty:
        return []
    up =part[part["window_start_s"] < 330]
    down = part[part["window_start_s"] >= 330]
    pairs = []
    for i, row in up.iterrows():
        if down.empty:
            break
        j = (down["dom_f"] - row["dom_f"]).abs().idxmin()
        if abs(down.at[j, "dom_f"] - row["dom_f"]) <= DF_MAX:
            pairs.append((i, j))
    return pairs


def explore_q3(frame: pd.DataFrame, kind: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    for fname, axes in (("ARz", "z"), ("ARyz", "yz")):
        cols = g.ar_columns(axes)
        for cond, part in frame.groupby("condition"):
            values = data.loc[part.index, cols].apply(pd.to_numeric, errors="coerce")
            for k in K_VALUES:
                avg = pd.concat([averaged(p, values, k) for _, p in part.groupby("source_record")])
                usable = avg.dropna()
                if len(usable) < 5:
                    continue
                std = usable.std(ddof=1).replace(0, 1.0)
                z = (avg - usable.mean()) / std

                def dist(i, j):
                    d = z.loc[i] - z.loc[j]
                    return float(np.sqrt((d ** 2).sum())) if d.notna().all() else np.nan

                within = {}
                for day, p in part.groupby("day"):
                    ok = p[avg.loc[p.index].notna().all(axis=1)]
                    within[day] = [dist(i, j) for i, j in pairs_within(ok)]
                    within[day] = [v for v in within[day] if not np.isnan(v)]
                ia, ib = part[part["day"] == "01"], part[part["day"] == "02"]
                ta = dict(zip(ia["window_start_s"].round(2), ia.index))
                tb = dict(zip(ib["window_start_s"].round(2), ib.index))
                cross = [dist(ta[t], tb[t]) for t in sorted(set(ta) & set(tb))]
                cross = [v for v in cross if not np.isnan(v)]
                if not cross or not within.get("01") or not within.get("02"):
                    continue
                for base, other in (("01", "02"), ("02", "01")):
                    w = np.array(within[base])
                    for thr in ("p95", "max"):
                        t1 = float(np.percentile(w, 95)) if thr == "p95" else float(w.max())
                        fa = float((np.array(cross) > t1).mean())
                        rows.append({
                            "kind": kind, "feature": fname, "condition": cond, "k": k,
                            "baseline_day": base, "threshold": thr,
                            "n_within": int(len(w)), "n_cross": int(len(cross)),
                            "within_median": float(np.median(w)), "cross_median": float(np.median(cross)),
                            "ratio_cross_within": float(np.median(cross) / np.median(w)),
                            "T": t1, "cross_day_fa": fa,
                            "factor_for_5pct": float(np.percentile(cross, 95) / t1),
                        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-by", choices=["bridge", "as"], default="bridge")
    args = parser.parse_args(argv)
    raw, deg = loader.load_pair(args.data_dir, "hbta")
    frame = prepare(raw, deg)
    if args.pair_by == "as":
        frame = attach_as(frame, args.data_dir / "hbta_as_dev.csv")
    q3 = []
    for kind, data in (("deg", deg), ("raw", raw)):
        f = frame.copy()
        if kind == "raw":
            f["held"] = frame["held"]          # same usable windows for a fair comparison
        q3 += explore_q3(f, kind, data)
    table = pd.DataFrame(q3)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "explore_truss_sm_q3.csv", index=False, encoding="utf-8-sig")
    summary = {
        "note": "개발용 탐색 (결과 열람 후). 성능 근거로 쓰지 않음",
        "files": {p.name: loader.file_digest(p) for p in sorted(Path(args.data_dir).glob("hbta_*.csv"))},
        "q1_usable": usable_ranges(frame),
        "q2_timing": timing(frame),
    }
    (args.out / "explore_truss_sm_q12.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"q3 rows {len(table)} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
