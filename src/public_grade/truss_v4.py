"""Truss validation v4 (prereg docs/AI2_PUBLIC_GRADE_PREREG_v4_TRUSS.md).

One-time evaluation of grade rule v3 ("이상의 확실성", unchanged) on data that
neither AI① nor AI② has seen:
  lumo        LUMO lattice tower (main): 31 nine-hour spans, horizontal x/y
  vanersborg  Vänersborg bascule truss bridge (auxiliary): 64 opening events

Pass rule (degraded data, section 7): false-alarm rate <= 5 % on the
false-alarm spans/events and danger detection >= 95 % on the danger
spans/events, where "abnormal" = grade 경고 or 위험 and only judged windows
(not held, persistence level available) count.

Each dataset is evaluated once. The summary file is never overwritten.

Usage:
  python -m src.public_grade.truss_v4 --data-dir data/raw/ai1_public_1.4 [--dataset lumo|vanersborg]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import grade_v3 as gv
from src.public_grade import grading as g
from src.public_grade import loader

PREREG = Path("docs/AI2_PUBLIC_GRADE_PREREG_v4_TRUSS.md")
OUT_DIR = Path("data/processed/public_grade/truss_v4")
SUMMARY_DIR = Path("results/public_grade")
STATUS = "사전 등록 v4 검증 (아직 보지 않은 데이터, 1회 실행)"

FA_MAX = 0.05
DETECT_MIN = 0.95
MIN_JUDGED = 200
ABNORMAL = ("경고", "위험")
KEY = ["experiment_id", "window_start_us"]
LABELS = ["source_record", "damage_stage", "label_status", "ai1_split", "cv_group"]

SPAN_HOURS = 9
LUMO_WINDOWS_PER_BLOCK = 18          # 599.78 s block, 60 s windows every 30 s from the block start (section 3)
MAX_SHIFTS = 3                    # section 4 fallback: up to 3 one-day shifts
BLOCK_S = 600                     # a window may start up to one block before the span start


class DeliveryError(ValueError):
    """The delivery does not match the preregistered composition (section 9)."""


# ------------------------------------------------------------------ design (sections 4, 5)

@dataclass(frozen=True)
class Span:
    name: str           # cv_group
    stage: str          # damage_stage
    split: str          # ai1_split
    role: str
    start: str          # UTC, "YYYY-MM-DD HH:MM"
    shift: int          # fallback direction: +1 = later (A, N1), -1 = earlier (B)


def _lumo_spans() -> list[Span]:
    n1 = [("N1_B1_0803", "train", "기준 적합", "2020-08-03 00:00"),
          ("N1_B1_0905", "train", "기준 적합", "2020-09-05 12:00"),
          ("N1_B2_0814", "ai2_calib", "정상 기준", "2020-08-14 12:00"),
          ("N1_B2_0825", "ai2_calib", "정상 기준", "2020-08-25 00:00"),
          ("N1_B2_0916", "ai2_calib", "정상 기준", "2020-09-16 00:00"),
          ("N1_B2_0927", "ai2_calib", "정상 기준", "2020-09-27 12:00"),
          ("N1_S_1006", "test", "같은 계절 오경보", "2020-10-06 00:00")]
    spans = [Span(n, "UDS", s, r, t, +1) for n, s, r, t in n1]
    pairs = [("N2", "UDS", "오경보 확인", "2020-10-29 00:00", "2020-11-08 12:00"),
             ("N3", "UDS", "오경보 확인", "2020-11-26 00:00", "2021-03-17 12:00"),
             ("N4", "UDS", "오경보 확인", "2021-04-22 00:00", "2021-05-03 12:00"),
             ("N5", "UDS", "오경보 확인", "2021-05-21 00:00", "2021-05-27 12:00"),
             ("N6", "UDS", "오경보 확인", "2021-06-16 00:00", "2021-06-24 12:00"),
             ("N7", "UDS", "오경보 확인", "2021-07-14 00:00", "2021-07-30 12:00"),
             ("DAM6_111", "DAM6_111", "위험 탐지", "2020-10-15 00:00", "2020-10-26 12:00"),
             ("DAM4_111", "DAM4_111", "위험 탐지", "2020-11-11 00:00", "2020-11-23 12:00"),
             ("DAM3_111", "DAM3_111", "위험 탐지", "2021-03-20 00:00", "2021-04-19 12:00"),
             ("DAM6_010", "DAM6_010", "작은 손상", "2021-05-06 00:00", "2021-05-18 12:00"),
             ("DAM4_010", "DAM4_010", "작은 손상", "2021-05-30 00:00", "2021-06-13 12:00"),
             ("DAM3_010", "DAM3_010", "작은 손상", "2021-06-27 00:00", "2021-07-11 12:00")]
    for name, stage, role, a, b in pairs:
        spans.append(Span(f"{name}_A", stage, "test", role, a, +1))
        spans.append(Span(f"{name}_B", stage, "test", role, b, -1))
    return spans


LUMO_SPANS = _lumo_spans()

VAN_N_PRE, VAN_N_POST = 57, 6
VAN_DUR = "2023_03_09T23_45_25"
VAN_ROLES = [(0, 20, "train", "PRE", "기준 적합"), (20, 40, "ai2_calib", "PRE", "정상 기준"),
             (40, 57, "test", "PRE", "오경보 확인")]


@dataclass(frozen=True)
class Design:
    name: str
    title: str
    ar_columns: tuple
    tac_columns: tuple
    sensors: tuple            # (sensor, axes) for the per-sensor AR distance in the reason
    step: float
    normal_stage: str
    names: dict
    fa_role: str = "오경보 확인"
    danger_role: str = "위험 탐지"


def _ar(axes: str) -> tuple:
    return tuple(f"{s}_ar{k}_{a}" for a in axes for s in ("acc01", "acc02") for k in range(1, 6))


DESIGNS = {
    "lumo": Design("lumo", "LUMO 격자 철탑 (주 평가)", _ar("xy"), ("tac_a_b_x", "tac_a_b_y"),
                   (("acc01", "xy"), ("acc02", "xy")), 30.0, "UDS",
                   {"ar": "AR 진동 특성 거리(acc01+acc02 합산, 수평 x·y)",
                    "tac": "가속도계 acc01(ML5, 5.00m)–가속도계 acc02(ML1, 8.95m) 전달 관계(1-TAC, x·y 평균)"}),
    "vanersborg": Design("vanersborg", "Vänersborg 개폐식 트러스교 (보조)", _ar("z"), ("tac_a_b_z",),
                         (("acc01", "z"), ("acc02", "z")), 10.24, "PRE",
                         {"ar": "AR 진동 특성 거리(acc01+acc02 합산, 수직)",
                          "tac": "가속도계 acc01(A1, 개폐 경간 1/3)–가속도계 acc02(A5, 균형추 트러스) 전달 관계(1-TAC, 수직)"}),
}
SENSOR_LABEL = {"lumo": {"acc01": "가속도계 acc01(ML5, 5.00m)", "acc02": "가속도계 acc02(ML1, 8.95m)"},
                "vanersborg": {"acc01": "가속도계 acc01(A1)", "acc02": "가속도계 acc02(A5)"}}


# ------------------------------------------------------------------ loading and checks

def result_path(data_dir: Path, dataset: str, kind: str) -> Path:
    return Path(data_dir) / f"{dataset}_anomaly_results_{kind}.csv"


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(data_dir: Path, design: Design) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames, digests = {}, {}
    need = KEY + LABELS + list(design.ar_columns) + list(design.tac_columns)
    for kind in ("raw", "deg"):
        path = result_path(data_dir, design.name, kind)
        if not path.is_file():
            raise DeliveryError(f"missing delivery file: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = [c for c in need + (["hold_main"] if kind == "deg" else []) if c not in frame]
        if missing:
            raise DeliveryError(f"{path.name}: missing columns {missing}")
        frames[kind] = frame.sort_values(["source_record", "window_start_us"]).reset_index(drop=True)
        digests[path.name] = md5(path)
    raw, deg = frames["raw"], frames["deg"]
    if len(raw) != len(deg):
        raise DeliveryError("raw/deg row count differs")
    for col in KEY + LABELS:
        if not raw[col].astype("string").fillna("").equals(deg[col].astype("string").fillna("")):
            raise DeliveryError(f"raw/deg differ in {col}")
    if deg.duplicated(KEY).any():
        raise DeliveryError("duplicate window keys")
    held = loader.held_from_flag(deg["hold_main"]).to_numpy()
    for frame in (raw, deg):
        frame["held"] = held                      # raw uses the degraded windows (section 8.2)
    return raw, deg, digests


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def span_shift(span: Span, starts_us: np.ndarray) -> int | None:
    """Fallback shift (days) that puts every window of the span inside it, else None."""
    t = np.asarray(starts_us, float) / 1e6
    for k in range(MAX_SHIFTS + 1):
        s = (_utc(span.start) + timedelta(days=k * span.shift)).timestamp()
        if np.all(t >= s - BLOCK_S) and np.all(t < s + SPAN_HOURS * 3600):
            return k * span.shift
    return None


def check_lumo(frame: pd.DataFrame) -> dict:
    groups = set(frame["cv_group"].astype(str))
    expected = {s.name for s in LUMO_SPANS}
    if groups != expected:
        raise DeliveryError(f"LUMO spans differ: missing {sorted(expected - groups)}, extra {sorted(groups - expected)}")
    shifts = {}
    for span in LUMO_SPANS:
        part = frame[frame["cv_group"] == span.name]
        for col, want in (("damage_stage", span.stage), ("ai1_split", span.split)):
            got = set(part[col].astype(str))
            if got != {want}:
                raise DeliveryError(f"{span.name}: {col} {sorted(got)} != {want}")
        per_block = part.groupby("source_record").size()
        if (per_block > LUMO_WINDOWS_PER_BLOCK).any():
            raise DeliveryError(f"{span.name}: more than {LUMO_WINDOWS_PER_BLOCK} windows in a block")
        k = span_shift(span, part["window_start_us"].to_numpy())
        if k is None:
            raise DeliveryError(f"{span.name}: windows outside the preregistered span (and its fallback shifts)")
        shifts[span.name] = k
    return shifts


def vanersborg_events(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """(event, ai1_split, damage_stage, role) in file-name time order (section 5)."""
    events = sorted(frame["source_record"].astype(str).str.replace(".csv", "", regex=False).unique())
    if len(events) != VAN_N_PRE + 1 + VAN_N_POST:
        raise DeliveryError(f"Vänersborg: {len(events)} events, expected {VAN_N_PRE + 1 + VAN_N_POST}")
    if events[VAN_N_PRE] != VAN_DUR:
        raise DeliveryError(f"Vänersborg: event {VAN_N_PRE + 1} is {events[VAN_N_PRE]}, expected {VAN_DUR}")
    plan = []
    for lo, hi, split, stage, role in VAN_ROLES:
        plan += [(e, split, stage, role) for e in events[lo:hi]]
    plan.append((VAN_DUR, "test", "DUR", "파단 중"))
    plan += [(e, "test", "POST", "위험 탐지") for e in events[VAN_N_PRE + 1:]]
    return plan


def check_vanersborg(frame: pd.DataFrame) -> dict:
    names = frame["source_record"].astype(str).str.replace(".csv", "", regex=False)
    for event, split, stage, _ in vanersborg_events(frame):
        part = frame[names == event]
        for col, want in (("damage_stage", stage), ("ai1_split", split)):
            got = set(part[col].astype(str))
            if got != {want}:
                raise DeliveryError(f"{event}: {col} {sorted(got)} != {want}")
    return {}


def roles(frame: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == "lumo":
        by_group = {s.name: s.role for s in LUMO_SPANS}
        return frame["cv_group"].astype(str).map(by_group)
    names = frame["source_record"].astype(str).str.replace(".csv", "", regex=False)
    by_event = {e: r for e, _, _, r in vanersborg_events(frame)}
    return names.map(by_event)


# ------------------------------------------------------------------ grading (section 6)

def indicators(frame: pd.DataFrame, design: Design) -> tuple[pd.DataFrame, dict]:
    """AR distance from the 기준 적합 windows and 1-TAC (mean over axes; NaN if any axis is missing)."""
    out = pd.DataFrame(index=frame.index)
    fit = frame["ai1_split"].eq("train") & ~frame["held"]
    out["ar"], info = g.mahalanobis(frame[fit], frame, list(design.ar_columns))
    for sensor, axes in design.sensors:
        cols = [f"{sensor}_ar{k}_{a}" for a in axes for k in range(1, 6)]
        out[f"ar_{sensor}"], _ = g.mahalanobis(frame[fit], frame, cols)
    tac = frame[list(design.tac_columns)].apply(pd.to_numeric, errors="coerce")
    out["tac"] = tac.mean(axis=1, skipna=False)
    return out, info


def grade_frame(frame: pd.DataFrame, design: Design, role: pd.Series) -> tuple[pd.DataFrame, dict]:
    ind, info = indicators(frame, design)
    work = frame.copy()
    for col in ind:
        work[col] = ind[col]
    held = work["held"]
    normal = work["ai1_split"].eq("ai2_calib") & work["damage_stage"].eq(design.normal_stage) & ~held
    levels, b = pd.DataFrame(index=work.index), {}
    for col in design.names:
        b[col] = gv.bounds(work.loc[normal, col])
        levels[col] = gv.level(work[col], b[col])
    levels[held.to_numpy()] = np.nan
    labels = SENSOR_LABEL[design.name]
    subs = {"ar": [(f"ar_{s}", labels[s]) for s, _ in design.sensors]}
    sub_b = {col: gv.bounds(work.loc[normal, col]) for col, _ in subs["ar"]}
    out = gv._finish(work, levels, design.names, held, role.to_numpy(), sequential=True, bounds_=b,
                     step=design.step, group_col="source_record", subs=subs, sub_bounds=sub_b)
    out["cv_group"] = work["cv_group"].to_numpy()
    out["status"] = STATUS
    out["judged"] = ~held.to_numpy() & out["sustained_level"].notna().to_numpy()
    out["abnormal"] = out["grade"].isin(ABNORMAL)
    meta = {"bounds": {k: vars(v) for k, v in b.items()}, "sub_bounds": {k: vars(v) for k, v in sub_b.items()},
            "ar_fit": info, "n_normal_reference": int(normal.sum())}
    return out, meta


# ------------------------------------------------------------------ metrics (sections 7, 8)

def rate(w: pd.DataFrame, mask) -> dict:
    sel = w[np.asarray(mask, bool)]
    judged = sel[sel["judged"]]
    n, k = int(len(judged)), int(judged["abnormal"].sum())
    return {"windows": int(len(sel)), "held": int((sel["grade"] == gv.HELD).sum()),
            "undecided": int((~sel["judged"] & (sel["grade"] != gv.HELD)).sum()),
            "judged": n, "abnormal": k, "rate": (k / n) if n else None}


def grade_shares(w: pd.DataFrame, mask) -> dict:
    sel = w[np.asarray(mask, bool) & w["judged"].to_numpy()]
    n = len(sel)
    return {gr: (float((sel["grade"] == gr).sum()) / n if n else None) for gr in gv.GRADES}


def auc(normal, danger) -> float | None:
    a = pd.to_numeric(pd.Series(normal), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(danger), errors="coerce").dropna().to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        return None
    greater = (b[:, None] > a[None, :]).mean()
    ties = (b[:, None] == a[None, :]).mean()
    return float(greater + 0.5 * ties)


def decide(fa: dict, det: dict) -> dict:
    fa_ok = fa["rate"] is not None and fa["rate"] <= FA_MAX
    det_ok = det["rate"] is not None and det["rate"] >= DETECT_MIN
    return {"false_alarm_pass": fa_ok, "detection_pass": det_ok, "pass": bool(fa_ok and det_ok),
            "tentative": bool(fa["judged"] < MIN_JUDGED or det["judged"] < MIN_JUDGED)}


def evaluate(w_deg: pd.DataFrame, w_raw: pd.DataFrame, design: Design) -> dict:
    role = w_deg["role"]
    fa_mask, det_mask = role.eq(design.fa_role), role.eq(design.danger_role)
    fa, det = rate(w_deg, fa_mask), rate(w_deg, det_mask)
    result = {"false_alarm": fa, "detection": det, **decide(fa, det)}
    by = "cv_group" if design.name == "lumo" else "source_record"
    result["by_group"] = {str(k): {"role": str(role[idx].iloc[0]), **rate(w_deg, w_deg.index.isin(idx)),
                                   "grades": grade_shares(w_deg, w_deg.index.isin(idx))}
                          for k, idx in w_deg.groupby(by).groups.items()}
    result["by_stage"] = {str(k): {**rate(w_deg, w_deg.index.isin(idx)),
                                   "grades": grade_shares(w_deg, w_deg.index.isin(idx))}
                          for k, idx in w_deg.groupby("damage_stage").groups.items()}
    # 8.2 raw on the windows judged in the degraded data
    deg_judged = w_deg["judged"].to_numpy()
    raw = w_raw.copy()
    raw["judged"] = raw["judged"].to_numpy() & deg_judged
    raw_fa, raw_det = rate(raw, fa_mask), rate(raw, det_mask)
    result["raw"] = {"false_alarm": raw_fa, "detection": raw_det, **decide(raw_fa, raw_det)}
    # 8.3 AUC per indicator on judged windows
    result["auc"] = {}
    for kind, w in (("deg", w_deg), ("raw", raw)):
        j = w["judged"].to_numpy()
        result["auc"][kind] = {col: auc(w.loc[fa_mask.to_numpy() & j, f"value_{col}"],
                                        w.loc[det_mask.to_numpy() & j, f"value_{col}"]) for col in design.names}
    if design.name == "lumo":                      # 8.1 same-season false alarm
        result["same_season_false_alarm"] = rate(w_deg, role.eq("같은 계절 오경보"))
        result["small_damage"] = rate(w_deg, role.eq("작은 손상"))
    else:                                           # 8.5 event level
        events = {}
        for name, part in w_deg.groupby("source_record"):
            events[str(name)] = {"role": str(part["role"].iloc[0]), "any_danger": bool((part["grade"] == "위험").any())}
        result["events_any_danger"] = {r: {"events": sum(1 for e in events.values() if e["role"] == r),
                                           "with_danger": sum(1 for e in events.values() if e["role"] == r and e["any_danger"])}
                                       for r in (design.fa_role, design.danger_role)}
        dur = w_deg[role.eq("파단 중")].sort_values("window_start_us")
        result["during_event_grades"] = dur["grade"].tolist()
    return result


# ------------------------------------------------------------------ run

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run(data_dir: Path, dataset: str, out_dir: Path = OUT_DIR, summary_dir: Path = SUMMARY_DIR,
        prereg: Path = PREREG) -> dict:
    design = DESIGNS[dataset]
    summary_path = Path(summary_dir) / f"truss_v4_{dataset}_summary.json"
    if summary_path.exists():
        raise SystemExit(f"{summary_path} exists: v4 is run once per dataset (prereg section 0)")
    raw, deg, digests = load(data_dir, design)
    shifts = check_lumo(deg) if dataset == "lumo" else check_vanersborg(deg)
    role = roles(deg, dataset)
    w_deg, meta_deg = grade_frame(deg, design, role)
    w_raw, meta_raw = grade_frame(raw, design, role)
    result = evaluate(w_deg, w_raw, design)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_deg.to_csv(out_dir / f"{dataset}_deg_grades.csv", index=False, encoding="utf-8-sig")
    w_raw.to_csv(out_dir / f"{dataset}_raw_grades.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset": dataset, "title": design.title, "status": STATUS,
        "prereg": str(prereg), "prereg_sha256": sha256(prereg) if Path(prereg).exists() else None,
        "code_commit": git_commit(), "inputs": digests, "span_shifts_days": shifts,
        "criteria": {"false_alarm_max": FA_MAX, "detection_min": DETECT_MIN, "abnormal": list(ABNORMAL),
                     "min_judged_for_final": MIN_JUDGED},
        "deg": meta_deg, "raw": meta_raw, "result": result,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dataset", choices=sorted(DESIGNS), action="append")
    args = ap.parse_args(argv)
    for dataset in args.dataset or sorted(DESIGNS):
        s = run(args.data_dir, dataset)
        r = s["result"]
        tag = "통과" if r["pass"] else "통과 못 함"
        if r["tentative"]:
            tag += " (잠정: 판정 Window 200개 미만)"
        print(f"[{s['title']}] {tag}")
        print(f"  오경보율 {_pct(r['false_alarm']['rate'])} (판정 {r['false_alarm']['judged']}, 기준 ≤ 5%)")
        print(f"  위험 탐지율 {_pct(r['detection']['rate'])} (판정 {r['detection']['judged']}, 기준 ≥ 95%)")
        print(f"  원본(비교): 오경보 {_pct(r['raw']['false_alarm']['rate'])}, 탐지 {_pct(r['raw']['detection']['rate'])}")
        print(f"  요약: {SUMMARY_DIR / f'truss_v4_{dataset}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
