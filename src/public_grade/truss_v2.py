"""Truss v2 normal/anomaly evaluation on HBTA sweep (SM) data.

Implements docs/AI2_PUBLIC_GRADE_PREREG_v2_TRUSS.md. Section numbers in the
comments refer to that document. Run once on AI1-public-1.2 after the
pre-registration is pushed:

  python -m src.public_grade.truss_v2 --data-dir data/raw/ai1_public_1.2 \
      --dev-dir data/raw/ai1_public_1.2_dev

`--dev-only` recomputes the fixed thresholds and the development transfer
numbers from AI1-public-1.2-dev without touching check data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import grading as g
from src.public_grade import loader

PREREG = Path("docs/AI2_PUBLIC_GRADE_PREREG_v2_TRUSS.md")
WINDOW_S = 10.24
T_DEG = 2.6795597336          # section 6
T_RAW = 2.0605781871          # section 8.2
T_TOLERANCE = 1e-9
QUANTILE = 99
PERSIST_M, PERSIST_N = 2, 3
FA_LIMIT, DETECT_MIN = 0.05, 0.95
MIN_FA_WINDOWS = 20
AS_ON_RMS = 1.0               # section 4: shaker running when AS RMS >= 1 m/s^2
LAGS = range(-3, 4)
AUC_Y_LOSS = 0.99

MAIN = "SM_P2_Z"
MAIN_BASELINE = "MVS_P2_UDS_SM_Z_01"
MAIN_FA = "MVS_P2_UDS_SM_Z_02"
DANGER = ("DS2", "DS8")
EXPECTED_GRADE = {"DS3": "주의", "DS5": "주의", "DS6": "주의",
                  "DS1": "경고", "DS4": "경고", "DS7": "경고",
                  "DS2": "위험", "DS8": "위험"}
SIDE_CONDITIONS = ("SM_P1_Y", "SM_P1_Z", "SM_P2_Y")
DEV_RECORDS = {
    "SM_P1_Y": ("MVS_P1_UDS_SM_Y_01", "MVS_P1_UDS_SM_Y_02"),
    "SM_P1_Z": ("MVS_P1_UDS_SM_Z_01", "MVS_P1_UDS_SM_Z_02"),
    "SM_P2_Y": ("MVS_P2_UDS_SM_Y_01", "MVS_P2_UDS_SM_Y_02"),
}
CHECK_RECORDS = (
    [MAIN_BASELINE, MAIN_FA] + [f"MVS_P2_DS{i}_SM_Z_01" for i in range(1, 9)]
    + [f"MVS_P1_DS{i}_SM_Y_01" for i in range(1, 9)]
    + [f"MVS_P1_DS{i}_SM_Z_01" for i in range(1, 9)]
    + [f"MVS_P2_DS{i}_SM_Y_01" for i in range(1, 9)]
)
AR_Z = g.ar_columns("z")
AR_YZ = g.ar_columns("yz")


class CheckError(ValueError):
    """Delivery or pre-registration mismatch: stop before evaluating."""


# ------------------------------------------------------------------ loading

@dataclass
class Delivery:
    raw: pd.DataFrame
    deg: pd.DataFrame
    shaker: pd.DataFrame
    digests: dict


def load(data_dir: Path, as_name: str) -> Delivery:
    raw, deg = loader.load_pair(data_dir, "hbta")
    shaker = loader.read_csv(Path(data_dir) / as_name)
    keys = ["experiment_id", "window_start_us"]
    if not shaker[keys].reset_index(drop=True).equals(raw[keys].reset_index(drop=True)):
        raise CheckError(f"{as_name}: keys or order differ from the result files")
    held = loader.held_from_flag(deg["hold_main"])          # section 2
    for frame in (raw, deg):
        frame["held"] = held.to_numpy()
        frame["t"] = pd.to_numeric(frame["window_start_s"]).round(2)
    shaker["t"] = pd.to_numeric(shaker["window_start_s"]).round(2)
    digests = {p.name: loader.file_digest(p) for p in sorted(Path(data_dir).glob("*.csv"))}
    return Delivery(raw, deg, shaker, digests)


def record(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    part = frame[frame["source_record"] == name]
    if part.empty:
        raise CheckError(f"missing record {name}")
    return part.set_index("t").sort_index()


# ------------------------------------------------------------------ indicator (5, 6)

def scale(baseline: pd.DataFrame, columns: list[str]) -> np.ndarray:
    usable = baseline.loc[~baseline["held"], columns].apply(pd.to_numeric, errors="coerce").dropna()
    s = usable.std(ddof=1).to_numpy(float).copy()
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return s


def distance(baseline: pd.DataFrame, target: pd.DataFrame, columns: list[str],
             shift_windows: int = 0) -> pd.Series:
    """d(t) on windows usable in both records, indexed by baseline time t."""
    tgt = target.copy()
    tgt.index = (tgt.index - shift_windows * WINDOW_S).round(2)
    common = baseline.index.intersection(tgt.index)
    b, x = baseline.loc[common], tgt.loc[common]
    usable = ~b["held"].to_numpy() & ~x["held"].to_numpy()
    bv = b.loc[usable, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    xv = x.loc[usable, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    d = np.sqrt(np.mean(((xv - bv) / scale(baseline, columns)) ** 2, axis=1))
    out = pd.Series(d, index=common[usable], name="d")
    return out[np.isfinite(out)]


def persist(d: pd.Series, threshold: float, m: int = PERSIST_M, n: int = PERSIST_N) -> pd.Series:
    """1 = 이상, 0 = 정상, NaN = 판정 불가 (fewer than n consecutive usable windows)."""
    t = d.index.to_numpy(float)
    over = (d.to_numpy() > threshold).astype(int)
    out = np.full(len(d), np.nan)
    for i in range(n - 1, len(d)):
        if np.all(np.abs(np.diff(t[i - n + 1:i + 1]) - WINDOW_S) < 1e-3):
            out[i] = float(over[i - n + 1:i + 1].sum() >= m)
    return pd.Series(out, index=d.index, name="flag")


def rate(flags: pd.Series) -> dict:
    v = flags.dropna()
    return {"n": int(len(v)), "anomaly": int(v.sum()), "rate": float(v.mean()) if len(v) else None,
            "undecided": int(flags.isna().sum())}


def auc(normal, danger) -> float | None:
    a, b = np.asarray(normal, float), np.asarray(danger, float)
    if not len(a) or not len(b):
        return None
    return float((b[:, None] > a[None, :]).mean() + 0.5 * (b[:, None] == a[None, :]).mean())


# ------------------------------------------------------------------ fixed thresholds (6, 8.2)

def dev_pooled(dev: Delivery, data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    pooled = []
    for first, second in DEV_RECORDS.values():
        a, b = record(data, first), record(data, second)
        pooled.append(distance(a, b, columns).to_numpy())
        pooled.append(distance(b, a, columns).to_numpy())
    return np.concatenate(pooled)


def check_thresholds(dev: Delivery) -> dict:
    out = {}
    for kind, data, fixed in (("deg", dev.deg, T_DEG), ("raw", dev.raw, T_RAW)):
        pooled = dev_pooled(dev, data, AR_Z)
        value = float(np.percentile(pooled, QUANTILE))
        out[kind] = {"recomputed": value, "fixed": fixed, "n": int(len(pooled))}
        if abs(value - fixed) > T_TOLERANCE:
            raise CheckError(f"{kind} threshold {value:.10f} differs from pre-registered {fixed:.10f}")
    return out


def dev_false_alarm(dev: Delivery) -> dict:
    """Section 8.3: false-alarm rate of the side conditions on development data (seen)."""
    out = {}
    for cond, (first, second) in DEV_RECORDS.items():
        d = distance(record(dev.deg, first), record(dev.deg, second), AR_Z)
        out[cond] = rate(persist(d, T_DEG))
    return out


# ------------------------------------------------------------------ alignment (4)

def best_lag(base_as: pd.DataFrame, target_as: pd.DataFrame) -> dict:
    """Lag (in windows) minimising the median |log f_base - log f_target| over
    windows where the shaker runs in both records. Correlation is not used: on a
    log sweep it is the same for every lag."""
    scores = {}
    for lag in LAGS:
        tgt = target_as.copy()
        tgt.index = (tgt.index - lag * WINDOW_S).round(2)
        common = base_as.index.intersection(tgt.index)
        b, x = base_as.loc[common], tgt.loc[common]
        on = (b["as_rms_m_s2"] >= AS_ON_RMS) & (x["as_rms_m_s2"] >= AS_ON_RMS)
        if on.sum() < 5:
            continue
        diff = np.abs(np.log(b.loc[on, "as_dom_freq_hz"]) - np.log(x.loc[on, "as_dom_freq_hz"]))
        scores[lag] = float(np.median(diff))
    lag = min(scores, key=lambda k: (scores[k], abs(k))) if scores else None
    return {"best_lag": lag, "median_abs_log_diff_by_lag": scores, "misaligned": lag not in (0, None)}


# ------------------------------------------------------------------ evaluation

def evaluate_condition(baseline_deg, baseline_raw, targets: dict, shaker_base, shaker: dict) -> dict:
    """targets: record name -> (deg frame, raw frame, damage_stage)."""
    rows, per_record, windows = {}, {}, []
    for name, (deg_t, raw_t, stage) in targets.items():
        align = best_lag(shaker_base, shaker[name])
        d = distance(baseline_deg, deg_t, AR_Z)
        flags = persist(d, T_DEG)
        per_record[name] = {"stage": stage, "alignment": align, "deg": rate(flags),
                            "held": int(deg_t["held"].sum()), "n_windows": int(len(deg_t)),
                            "d_median": float(d.median()) if len(d) else None}
        if align["misaligned"]:
            shifted = persist(distance(baseline_deg, deg_t, AR_Z, align["best_lag"]), T_DEG)
            per_record[name]["deg_shifted_reference"] = rate(shifted)
        d_raw = distance(baseline_raw, raw_t, AR_Z)
        per_record[name]["raw"] = rate(persist(d_raw, T_RAW))
        rows[name] = {
            "d_deg": d, "flags": flags, "d_raw": d_raw,
            "d_raw_yz": distance(baseline_raw, raw_t, AR_YZ),
            "d_deg_yz": distance(baseline_deg, deg_t, AR_YZ),
        }
        windows.append(pd.DataFrame({"source_record": name, "damage_stage": stage, "t": d.index,
                                     "d_deg": d.to_numpy(), "flag": flags.to_numpy()}))
    return {"rows": rows, "per_record": per_record,
            "windows": pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()}


def pooled(rows: dict, per_record: dict, key: str, stages) -> pd.Series:
    parts = [rows[n][key] for n, r in per_record.items() if r["stage"] in stages]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def main_evaluation(check: Delivery) -> dict:
    base_deg, base_raw = record(check.deg, MAIN_BASELINE), record(check.raw, MAIN_BASELINE)
    shaker_base = record(check.shaker, MAIN_BASELINE)
    names = [MAIN_FA] + [f"MVS_P2_DS{i}_SM_Z_01" for i in range(1, 9)]
    targets, shaker = {}, {}
    for name in names:
        deg_t = record(check.deg, name)
        stage = str(deg_t["damage_stage"].iloc[0])
        targets[name] = (deg_t, record(check.raw, name), stage)
        shaker[name] = record(check.shaker, name)
    res = evaluate_condition(base_deg, base_raw, targets, shaker_base, shaker)
    rows, per = res["rows"], res["per_record"]

    fa = rate(rows[MAIN_FA]["flags"])
    danger = rate(pooled(rows, per, "flags", DANGER))
    c1 = {**fa, "limit": FA_LIMIT, "pass": fa["rate"] is not None and fa["rate"] <= FA_LIMIT,
          "provisional": fa["n"] < MIN_FA_WINDOWS}
    c2 = {**danger, "min": DETECT_MIN, "pass": danger["rate"] is not None and danger["rate"] >= DETECT_MIN,
          "by_stage": {s: per[f"MVS_P2_{s}_SM_Z_01"]["deg"] for s in DANGER}}

    # 8.1 separability, 8.2 cause
    normal_d = rows[MAIN_FA]["d_deg"]
    danger_d = pooled(rows, per, "d_deg", DANGER)
    oracle_t = float(np.percentile(normal_d, 95)) if len(normal_d) else None
    raw_fa = rate(persist(rows[MAIN_FA]["d_raw"], T_RAW))
    raw_danger = rate(pooled({n: {"f": persist(r["d_raw"], T_RAW)} for n, r in rows.items()}, per, "f", DANGER))
    raw_pass = (raw_fa["rate"] is not None and raw_fa["rate"] <= FA_LIMIT
                and raw_danger["rate"] is not None and raw_danger["rate"] >= DETECT_MIN)
    raw_yz_auc = auc(rows[MAIN_FA]["d_raw_yz"], pooled(rows, per, "d_raw_yz", DANGER))
    passed = c1["pass"] and c2["pass"]
    if passed:
        cause = None
    elif raw_pass:
        cause = "센서 잡음 (열화본 실패, 원본 z 통과)"
    elif raw_yz_auc is not None and raw_yz_auc >= AUC_Y_LOSS:
        cause = "y축 손실 (원본 z 실패, 원본 y+z AUC ≥ 0.99)"
    else:
        cause = "하루치 정상 기준 또는 데이터 한계 (원본 y+z AUC < 0.99)"

    # 8.5 grade description
    medians = {s: per[f"MVS_P2_{s}_SM_Z_01"]["d_median"] for s in EXPECTED_GRADE}
    medians["UDS (_02)"] = per[MAIN_FA]["d_median"]
    return {
        "verdict": "통과 (공개 데이터 SM P2 Z, 하루치 정상 기준. API 연결 안 함)" if passed
        else "실패: Truss는 공개 데이터 기준으로 '기준 대비 이상 여부'를 출력할 수 없음",
        "conditions": {"1_false_alarm": c1, "2_danger_detection": c2},
        "per_record": per,
        "separability": {
            "auc_deg_z": auc(normal_d, danger_d),
            "oracle_threshold_p95_of_02": oracle_t,
            "oracle_detection": float((danger_d > oracle_t).mean()) if oracle_t is not None and len(danger_d) else None,
            "note": "확인용 정상 데이터로 정한 상한값. 성능 주장에 쓰지 않음",
        },
        "cause_analysis": {
            "raw_z": {"false_alarm": raw_fa, "danger": raw_danger, "pass": raw_pass},
            "raw_yz_auc": raw_yz_auc, "cause": cause,
        },
        "grade_description": {"stage_median_d": medians, "expected_grade": EXPECTED_GRADE,
                              "note": "서술만. 통과 판정 없음"},
        "windows": res["windows"],
    }


def side_evaluation(check: Delivery, dev: Delivery) -> dict:
    """Section 8.3: other conditions, baseline = development _01 of the condition."""
    dev_fa = dev_false_alarm(dev)
    out = {}
    for cond in SIDE_CONDITIONS:
        base_name = DEV_RECORDS[cond][0]
        pos, axis = cond.split("_")[1], cond.split("_")[2]
        names = [f"MVS_{pos}_DS{i}_SM_{axis}_01" for i in range(1, 9)]
        targets, shaker = {}, {}
        for name in names:
            deg_t = record(check.deg, name)
            targets[name] = (deg_t, record(check.raw, name), str(deg_t["damage_stage"].iloc[0]))
            shaker[name] = record(check.shaker, name)
        res = evaluate_condition(record(dev.deg, base_name), record(dev.raw, base_name),
                                 targets, record(dev.shaker, base_name), shaker)
        rows, per = res["rows"], res["per_record"]
        dev_normal = distance(record(dev.deg, base_name), record(dev.deg, DEV_RECORDS[cond][1]), AR_YZ)
        out[cond] = {
            "danger_detection": rate(pooled(rows, per, "flags", DANGER)),
            "by_stage": {r["stage"]: r["deg"] for r in per.values()},
            "false_alarm_dev_seen": dev_fa[cond],
            "auc_deg_yz_reference": auc(dev_normal, pooled(rows, per, "d_deg_yz", DANGER)),
            "alignment": {n: r["alignment"]["best_lag"] for n, r in per.items()},
        }
    return out


def ai1_reference(check: Delivery) -> dict:
    out = {}
    for name, part in check.deg.groupby("source_record"):
        pred = part["prediction"].astype(str)
        decided = pred[pred.isin(["Anomaly", "Normal"])]
        out[name] = {"anomaly_rate": float((decided == "Anomaly").mean()) if len(decided) else None,
                     "n_decided": int(len(decided))}
    return out


# ------------------------------------------------------------------ run

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--out-json", type=Path, default=Path("results/public_grade/truss_v2_summary.json"))
    parser.add_argument("--out-windows", type=Path, default=Path("data/processed/public_grade/truss_v2_windows.csv"))
    args = parser.parse_args(argv)

    dev = load(args.dev_dir, "hbta_as_dev.csv")
    thresholds = check_thresholds(dev)
    print(f"T 재계산: deg {thresholds['deg']['recomputed']:.10f} (고정 {T_DEG}), "
          f"raw {thresholds['raw']['recomputed']:.10f} (고정 {T_RAW})")
    if args.dev_only:
        for cond, r in dev_false_alarm(dev).items():
            print(f"  개발용 {cond}: 오경보 {r['anomaly']}/{r['n']} (판정 불가 {r['undecided']})")
        return 0
    if args.data_dir is None:
        parser.error("--data-dir is required unless --dev-only")

    check = load(args.data_dir, "hbta_as.csv")
    missing = sorted(set(CHECK_RECORDS) - set(check.deg["source_record"]))
    if missing:
        raise CheckError(f"check delivery misses {missing}")
    main_res = main_evaluation(check)
    windows = main_res.pop("windows")
    summary = {
        "prereg": str(PREREG), "prereg_sha256": sha256(PREREG) if PREREG.exists() else None,
        "code_commit": git_commit(), "inputs": {"check": check.digests, "dev": dev.digests},
        "thresholds": thresholds, "main": main_res,
        "side": side_evaluation(check, dev), "ai1_v1_reference": ai1_reference(check),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1))
    args.out_windows.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.out_windows, index=False, encoding="utf-8-sig")
    c = main_res["conditions"]
    print(f"[주 평가 SM P2 Z] 오경보 {c['1_false_alarm']['anomaly']}/{c['1_false_alarm']['n']} "
          f"= {c['1_false_alarm']['rate']}, 위험 탐지 {c['2_danger_detection']['anomaly']}/"
          f"{c['2_danger_detection']['n']} = {c['2_danger_detection']['rate']}")
    print(f"판정: {main_res['verdict']}")
    if main_res["cause_analysis"]["cause"]:
        print(f"원인: {main_res['cause_analysis']['cause']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
