"""AI② risk grade v3 (plan A): grade every window of the AI① public datasets.

Rule: docs/AI2_GRADE_RULE_v3.md. Meaning of a grade = how certain it is that
the window is NOT like the normal reference ("이상의 확실성"), not damage
severity. Boundaries come from normal-state windows only; damage labels are
used only to tabulate the result.

All AI① public data has already been seen (v1, v2, exploration), so every
grade produced here carries the status "이미 본 데이터로 판정, 검증 아님".

Usage:
  python -m src.public_grade.grade_v3 --out-dir data/processed/public_grade/grade_v3 \
      --summary results/public_grade/grade_v3_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import grading as g
from src.public_grade import loader
from src.public_grade import truss_v2 as tv

GRADES = ("정상", "주의", "경고", "위험")
HELD, UNDECIDED = "판정 보류", "판정 불가"
STATUS = "이미 본 데이터로 판정, 검증 아님"
Q_LOW = 95

DIR_11 = Path("data/raw/ai1_public_1.1")
DIR_12 = Path("data/raw/ai1_public_1.2")
DIR_12_DEV = Path("data/raw/ai1_public_1.2_dev")

# SM (time-aligned AR difference): normal day-to-day differences of the
# development conditions (204 values, same set as Truss v2 section 6).
SM_BOUNDS = {"deg": (1.2336279781, 3.5955651454), "raw": (1.6878412257, 2.4703492908)}


# ------------------------------------------------------------------ core rule

@dataclass
class Bounds:
    low: float          # p95 of the normal reference
    high: float         # max of the normal reference
    n: int


def bounds(values) -> Bounds:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        raise ValueError("empty normal reference")
    return Bounds(float(np.percentile(v, Q_LOW)), float(v.max()), int(len(v)))


def level(values: pd.Series, b: Bounds) -> pd.Series:
    """0: <= p95 of normal, 1: above p95, 2: above the largest normal value. NaN if missing."""
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.where(v > b.high, 2.0, np.where(v > b.low, 1.0, 0.0)), index=values.index)
    out[v.isna()] = np.nan
    return out


def sustained(levels: pd.Series, times: pd.Series, groups: pd.Series, step: float) -> pd.Series:
    """Level reached in at least 2 of the last 3 consecutive windows (2nd largest of 3).
    NaN when the two previous windows are missing or not consecutive."""
    out = pd.Series(np.nan, index=levels.index)
    frame = pd.DataFrame({"lv": levels, "t": times, "grp": groups})
    for _, part in frame.groupby("grp", sort=False):
        part = part.sort_values("t")
        t, lv, idx = part["t"].to_numpy(float), part["lv"].to_numpy(float), part.index
        for i in range(2, len(part)):
            span, vals = t[i - 2:i + 1], lv[i - 2:i + 1]
            if np.all(np.abs(np.diff(span) - step) < 1e-3 * step) and not np.isnan(vals).any():
                out[idx[i]] = float(np.sort(vals)[-2])
    return out


def grade_sequential(window_level: pd.Series, sustained_level: pd.Series) -> pd.Series:
    """위험: above normal max in >= 2 of 3 consecutive windows; 경고: above p95 in >= 2 of 3;
    주의: above p95 now but not sustained (or no history); 정상: within p95."""
    out = pd.Series(UNDECIDED, index=window_level.index, dtype=object)
    e, s = window_level, sustained_level
    out[e == 0] = "정상"
    out[e >= 1] = "주의"
    out[s >= 1] = "경고"
    out[s >= 2] = "위험"
    return out


def grade_single(levels: pd.DataFrame) -> pd.Series:
    """One window per event (Old ADA): agreement of indicators replaces persistence.
    위험: >= 2 indicators above normal max; 경고: one above normal max or >= 2 above p95;
    주의: one above p95; 정상: all within p95."""
    above_low = (levels >= 1).sum(axis=1)
    above_high = (levels >= 2).sum(axis=1)
    out = pd.Series("정상", index=levels.index, dtype=object)
    out[above_low >= 1] = "주의"
    out[(above_high >= 1) | (above_low >= 2)] = "경고"
    out[above_high >= 2] = "위험"
    out[levels.isna().all(axis=1)] = UNDECIDED
    return out


def recent_counts(window_level: pd.Series, times: pd.Series, groups: pd.Series, step: float) -> pd.DataFrame:
    """For the last 3 consecutive windows: how many are above p95 (n_low) and above
    the normal max (n_high). NaN when the two previous windows are not available."""
    out = pd.DataFrame(np.nan, index=window_level.index, columns=["n_low", "n_high"])
    frame = pd.DataFrame({"lv": window_level, "t": times, "grp": groups})
    for _, part in frame.groupby("grp", sort=False):
        part = part.sort_values("t")
        t, lv, idx = part["t"].to_numpy(float), part["lv"].to_numpy(float), part.index
        for i in range(2, len(part)):
            span, vals = t[i - 2:i + 1], lv[i - 2:i + 1]
            if np.all(np.abs(np.diff(span) - step) < 1e-3 * step) and not np.isnan(vals).any():
                out.loc[idx[i]] = [float((vals >= 1).sum()), float((vals >= 2).sum())]
    return out


def _num(x: float) -> str:
    return f"{x:.3g}"


def indicator_text(name: str, value, lv, b: Bounds) -> str:
    """One indicator with its value and the boundary numbers it was compared with."""
    if lv is None or (isinstance(lv, float) and np.isnan(lv)):
        return f"{name} 값 없음"
    v = _num(float(value))
    if lv >= 2:
        return f"{name} {v}: 위험 경계 {_num(b.high)} 초과"
    if lv >= 1:
        return f"{name} {v}: 정상 상한 {_num(b.low)} 초과 (위험 경계 {_num(b.high)} 이하)"
    return f"{name} {v} (정상 상한 {_num(b.low)} 이하)"


def reason(grade: str, values: dict, levels: dict, names: dict, bounds_: dict,
           counts: dict | None, sequential: bool) -> str:
    """판정 사유: which indicator, its value, the boundary numbers, and the persistence rule."""
    if grade == HELD:
        return "판정 보류: AI① 보류 표시 (진동이 센서 잡음 수준이거나 센서 값 없음)"
    if grade == UNDECIDED:
        return "판정 불가: 지표 값 없음"
    texts = {c: indicator_text(names[c], values[c], levels[c], bounds_[c]) for c in names}
    over = [texts[c] for c in names if not np.isnan(levels[c]) and levels[c] >= 1]
    if grade == "정상" and not over:
        return "모든 지표 정상 범위: " + ", ".join(texts.values())
    head = " / ".join(over) if over else "이번 Window는 모든 지표 정상 범위"
    if not sequential:
        n_low = sum(1 for c in names if not np.isnan(levels[c]) and levels[c] >= 1)
        n_high = sum(1 for c in names if not np.isnan(levels[c]) and levels[c] >= 2)
        return (f"{head} / 지표 일치: 정상 상한 초과 {n_low}개, 위험 경계 초과 {n_high}개 "
                f"(경고: 위험 경계 1개 또는 상한 2개 이상, 위험: 위험 경계 2개)")
    if counts is None or np.isnan(counts["n_low"]):
        tail = "앞 Window 2개가 없어 지속 여부를 판단할 수 없음"
    elif grade == "위험":
        tail = f"최근 3개 Window 중 {int(counts['n_high'])}개가 위험 경계 초과 (위험 조건: 2개 이상)"
    elif grade == "경고":
        tail = f"최근 3개 Window 중 {int(counts['n_low'])}개가 정상 상한 초과 (경고 조건: 2개 이상)"
    else:
        tail = f"지속 아님: 최근 3개 Window 중 {int(counts['n_low'])}개만 정상 상한 초과"
    return f"{head} / {tail}"


# ------------------------------------------------------------------ datasets

@dataclass
class Result:
    dataset: str
    bridge: str
    kind: str
    windows: pd.DataFrame
    bounds: dict
    reference: str
    notes: list = field(default_factory=list)


def _finish(frame, levels, names, held, role, sequential, bounds_, step=None, group_col="source_record"):
    out = frame[["experiment_id", "window_start_us", "source_record", "damage_stage",
                 "label_status", "ai1_split"]].copy()
    for col in levels.columns:
        out[f"value_{col}"] = frame[col] if col in frame else np.nan
        out[f"level_{col}"] = levels[col]
        out[f"bound_normal_{col}"] = bounds_[col].low
        out[f"bound_danger_{col}"] = bounds_[col].high
    e = levels.max(axis=1, skipna=True)
    counts = None
    if sequential:
        times = pd.to_numeric(frame["window_start_us"]) / 1e6
        s = sustained(e, times, frame[group_col], step)
        counts = recent_counts(e, times, frame[group_col], step)
        grade = grade_sequential(e, s)
        out["window_level"], out["sustained_level"] = e, s
    else:
        grade = grade_single(levels)
        out["window_level"], out["sustained_level"] = e, np.nan
    grade[held.to_numpy()] = HELD
    out["grade"] = grade
    values = frame.reindex(columns=list(levels.columns))
    out["reason"] = [
        reason(gr, values.loc[i].to_dict(), levels.loc[i].to_dict(), names, bounds_,
               None if counts is None else counts.loc[i].to_dict(), sequential)
        for i, gr in zip(frame.index, grade.to_numpy())
    ]
    out["role"] = role
    out["status"] = STATUS
    return out


def box(kind: str) -> Result:
    raw, deg = loader.load_pair(DIR_11, "ponte_moesa")
    f = deg if kind == "deg" else raw
    normal = (f["damage_stage"] == "DS2") & (f["ai1_split"] != "test")
    levels, b = pd.DataFrame(index=f.index), {}
    names = {"tac_4_2_z": "1-TAC 4-2", "tac_2_3_z": "1-TAC 2-3"}
    holds = {"tac_4_2_z": f["held_main"], "tac_2_3_z": f["held_session"]}
    for col in names:
        ref = f.loc[normal & ~holds[col], col]
        b[col] = bounds(ref)
        lv = level(f[col], b[col])
        lv[holds[col].to_numpy()] = np.nan
        levels[col] = lv
    held = levels.isna().all(axis=1)
    role = np.where(normal, "정상 기준", "판정")
    w = _finish(f, levels, names, held, role, sequential=True, bounds_=b, step=30.0)
    return Result("ponte_moesa", "Box", kind, w, {k: vars(v) for k, v in b.items()},
                  "DS2 train·val·ai2_calib·session (11/26 저녁, 11/27 낮 포함), 보류 제외",
                  ["Window 60초, 30초 이동", "AR 계수가 전달 파일에 없어 1-TAC 두 쌍만 사용"])


def hbta_nm(kind: str) -> Result:
    raw, deg = loader.load_pair(DIR_11, "hbta")
    f = (deg if kind == "deg" else raw).copy()
    held = f["held_main"].copy()
    fit = (f["ai1_split"] == "train") & ~held
    f["ar_z"], _ = g.mahalanobis(f[fit], f, g.ar_columns("z"))
    normal = (f["damage_stage"] == "UDS") & f["ai1_split"].isin(["val", "ai2_calib"]) & ~held
    names = {"ar_z": "AR z 거리", "tac_a_b_z": "1-TAC z"}
    levels, b = pd.DataFrame(index=f.index), {}
    for col in names:
        b[col] = bounds(f.loc[normal, col])
        levels[col] = level(f[col], b[col])
    levels[held.to_numpy()] = np.nan
    role = np.where(f["ai1_split"].eq("train") & f["damage_stage"].eq("UDS"), "기준 적합",
                    np.where(normal, "정상 기준", "판정"))
    w = _finish(f, levels, names, held, role, sequential=True, bounds_=b, step=tv.WINDOW_S)
    return Result("hbta_nm", "Truss", kind, w, {k: vars(v) for k, v in b.items()},
                  "UDS 두 날짜(9/22, 9/27) val·ai2_calib. AR 평균·공분산은 AI① train",
                  ["노이즈 가진(NM) 10개 측정", "1-TAC는 Window 3개가 같은 값을 공유"])


def hbta_sm(kind: str) -> Result:
    check = tv.load(DIR_12, "hbta_as.csv")
    dev = tv.load(DIR_12_DEV, "hbta_as_dev.csv")
    data = {"deg": (check.deg, dev.deg), "raw": (check.raw, dev.raw)}[kind]
    b = Bounds(*SM_BOUNDS[kind], 204)
    plans = [("SM_P2_Z", data[0], "MVS_P2_UDS_SM_Z_01")]
    plans += [(cond, data[1], first) for cond, (first, _) in tv.DEV_RECORDS.items()]
    rows = []
    for cond, base_src, base_name in plans:
        base = tv.record(base_src, base_name)
        targets = []
        for src in (data[0], data[1]):
            for name in sorted(src.loc[src["condition"] == cond, "source_record"].unique()):
                if name != base_name:
                    targets.append((src, name))
        for src, name in targets:
            tgt = tv.record(src, name)
            d = tv.distance(base, tgt, tv.AR_Z)
            part = tgt.reset_index().copy()
            part["t_key"] = part["t"].round(2)
            part["sm_d"] = part["t_key"].map(d)
            part["baseline"] = base_name
            rows.append(part)
    f = pd.concat(rows, ignore_index=True)
    names = {"sm_d": "같은 시각 AR 차이"}
    levels = pd.DataFrame({"sm_d": level(f["sm_d"], b)})
    held = f["held"].astype(bool)
    is_dev = f["source_record"].isin([r for pair in tv.DEV_RECORDS.values() for r in pair])
    role = np.where(is_dev, "판정 (기준값 산출에 쓴 개발용 정상)", "판정")
    w = _finish(f, levels, names, held, role, sequential=True, bounds_={"sm_d": b}, step=tv.WINDOW_S)
    w["baseline"] = f["baseline"].to_numpy()
    w["condition"] = f["condition"].to_numpy()
    return Result("hbta_sm", "Truss", kind, w, {"sm_d": vars(b)},
                  "개발용 SM 3개 조건(P1 Y, P1 Z, P2 Y)의 날짜 간 차이 204개 (Truss v2와 같은 값)",
                  ["스윕 가진(SM). 정상 기준 측정(각 조건 _01)은 판정하지 않음",
                   "Truss v2에서 이 지표는 손상을 거의 구분하지 못함(AUC 0.60)"])


def old_ada(kind: str) -> Result:
    raw, deg = loader.load_pair(DIR_11, "old_ada")
    f = (deg if kind == "deg" else raw).copy()
    cols = [c for c in f.columns if c.startswith(("acc01_ar", "acc02_ar")) and c.endswith("_z")]
    train = (f["ai1_split"] == "train") & ~f["held_main"]
    f["ar_z"], _ = g.standardized_euclid(f[train], f, cols)
    # leave-one-out distance for each training run (out-of-fit reference)
    for idx in f.index[train]:
        others = f[train & (f.index != idx)]
        f.loc[idx, "ar_z"] = g.standardized_euclid(others, f.loc[[idx]], cols)[0].iloc[0]
    normal = (f["damage_stage"] == "INT") & f["ai1_split"].isin(["train", "ai2_calib"]) & ~f["held_main"]
    names = {"ar_z": "AR z 거리", "tac_a_b_z": "1-TAC z"}
    levels, b = pd.DataFrame(index=f.index), {}
    for col in names:
        b[col] = bounds(f.loc[normal, col])
        levels[col] = level(f[col], b[col])
    held = f["held_main"].copy()
    levels[held.to_numpy()] = np.nan
    role = np.where(normal, "정상 기준", "판정")
    w = _finish(f, levels, names, held, role, sequential=False, bounds_=b)
    w["speed_kmh"] = f["speed_kmh"].to_numpy()
    return Result("old_ada", "점검용", kind, w, {k: vars(v) for k, v in b.items()},
                  "INT 40km/h train 4회(한 회씩 빼고 거리 계산)·ai2_calib 3회",
                  ["주행당 Window 1개라 연속 조건 대신 두 지표의 일치로 등급", f"AR 계수 {len(cols)}개 사용"])


# ------------------------------------------------------------------ summary

def distribution(w: pd.DataFrame, by: list[str]) -> list[dict]:
    rows = []
    for key, part in w.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        counts = part["grade"].value_counts()
        decided = part["grade"].isin(GRADES)
        n = int(decided.sum())
        row = dict(zip(by, key))
        row.update({"n_windows": int(len(part)), "n_graded": n,
                    "held": int((part["grade"] == HELD).sum()),
                    "undecided": int((part["grade"] == UNDECIDED).sum())})
        for gr in GRADES:
            row[gr] = round(float(counts.get(gr, 0)) / n, 4) if n else None
        row["주의 이상"] = round(float(part.loc[decided, "grade"].isin(GRADES[1:]).mean()), 4) if n else None
        rows.append(row)
    return rows


BUILDERS = {"ponte_moesa": box, "hbta_nm": hbta_nm, "hbta_sm": hbta_sm, "old_ada": old_ada}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/public_grade/grade_v3"))
    parser.add_argument("--summary", type=Path, default=Path("results/public_grade/grade_v3_summary.json"))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"rule": "docs/AI2_GRADE_RULE_v3.md", "meaning": "이상의 확실성 (손상 심각도 아님)",
               "status": STATUS, "code_commit": tv.git_commit(), "datasets": {}}
    for name, build in BUILDERS.items():
        for kind in ("deg", "raw"):
            res = build(kind)
            w = res.windows
            w.to_csv(args.out_dir / f"{name}_{kind}_grades.csv", index=False, encoding="utf-8-sig")
            by = ["condition", "role", "damage_stage"] if name == "hbta_sm" else ["role", "damage_stage", "ai1_split"]
            summary["datasets"][f"{name}_{kind}"] = {
                "bridge": res.bridge, "reference": res.reference, "bounds": res.bounds,
                "notes": res.notes, "distribution": distribution(w, by),
            }
            graded = w["grade"].value_counts().to_dict()
            print(f"{name:12s} {kind}: {len(w)} windows {graded}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(tv._jsonable(summary), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
