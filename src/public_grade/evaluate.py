"""Grade evaluation and pass conditions (prereg 4-8)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.public_grade import grading as g

MAIN_LABEL = "확실"
MIN_WINDOWS = 30
FA_LIMIT = 0.05
DETECTION_MIN = 0.95


@dataclass(frozen=True)
class BridgeSpec:
    name: str
    normal_stage: str
    stage_grade: dict
    danger_stages: tuple
    fold_column: str | None
    extent_pairs: tuple = field(default_factory=tuple)


BOX = BridgeSpec(
    name="ponte_moesa",
    normal_stage="DS2",
    stage_grade={"DS3": "주의", "DS5": "경고", "DS7": "위험"},
    danger_stages=("DS7",),
    fold_column="cv_group",
)

TRUSS = BridgeSpec(
    name="hbta",
    normal_stage="UDS",
    stage_grade={
        "DS3": "주의", "DS5": "주의", "DS6": "주의",
        "DS1": "경고", "DS4": "경고", "DS7": "경고",
        "DS2": "위험", "DS8": "위험",
    },
    danger_stages=("DS2", "DS8"),
    fold_column=None,
    extent_pairs=(("DS2", "DS1"), ("DS4", "DS3"), ("DS7", "DS5"), ("DS7", "DS6")),
)


def _f(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def prepare(frame: pd.DataFrame, values: pd.Series, held_column: str = "held_main") -> pd.DataFrame:
    out = frame.copy()
    out["value"] = pd.to_numeric(values, errors="coerce")
    out["held"] = out[held_column].astype(bool)
    out["valid"] = ~out["held"] & out["value"].notna()
    return out


def _main_damage(frame: pd.DataFrame, spec: BridgeSpec) -> pd.DataFrame:
    mask = (
        (frame["ai1_split"] == "test")
        & (frame["label_status"] == MAIN_LABEL)
        & frame["damage_stage"].isin(spec.stage_grade)
    )
    return frame[mask]


def _normal_test(frame: pd.DataFrame, spec: BridgeSpec) -> pd.DataFrame:
    mask = (
        (frame["ai1_split"] == "test")
        & (frame["label_status"] == MAIN_LABEL)
        & (frame["damage_stage"] == spec.normal_stage)
    )
    return frame[mask]


def _fit_upper(damage: pd.DataFrame, spec: BridgeSpec):
    grades = damage["damage_stage"].map(spec.stage_grade)
    pooled = {grade: damage.loc[grades == grade, "value"] for grade in ("주의", "경고", "위험")}
    return g.upper_thresholds(pooled)


def _counts(frame: pd.DataFrame, stage_mask) -> dict:
    part = frame[stage_mask]
    total = int(len(part))
    held = int(part["held"].sum())
    valid = int(part["valid"].sum())
    return {
        "total": total,
        "held": held,
        "held_ratio": held / total if total else None,
        "missing_value": int((~part["held"] & part["value"].isna()).sum()),
        "valid": valid,
        "provisional": valid < MIN_WINDOWS,
    }


def evaluate_bridge(frame: pd.DataFrame, spec: BridgeSpec) -> tuple[dict, pd.DataFrame]:
    """Evaluate one indicator on one bridge. frame must come from prepare()."""
    valid = frame[frame["valid"]]
    calib = valid[valid["ai1_split"] == "ai2_calib"]
    t0, t1 = g.lower_thresholds(calib["value"])
    normal = _normal_test(valid, spec)
    damage = _main_damage(valid, spec)

    t2, t3, t4, grade_medians = _fit_upper(damage, spec)
    final = (t0, t1, t2, t3, t4)

    if spec.fold_column:
        units = sorted(damage[spec.fold_column].dropna().unique())
        folds = [(f"{spec.fold_column}={u}", damage[spec.fold_column] == u) for u in units]
    else:
        units = sorted(damage["damage_stage"].unique())
        folds = [(f"without {u}", damage["damage_stage"] == u) for u in units]

    cv_grade = pd.Series(index=damage.index, dtype=object)
    cv_score = pd.Series(index=damage.index, dtype=float)
    cv_basis = pd.Series(index=damage.index, dtype=object)
    fold_results = []
    generalization = []
    for name, eval_mask in folds:
        evaluated = damage[eval_mask]
        fitted = damage[~eval_mask]
        f2, f3, f4, f_medians = _fit_upper(fitted, spec)
        fold_t = (t0, t1, f2, f3, f4)
        fold_valid = g.thresholds_valid(fold_t)
        fold_results.append({
            "fold": name, "thresholds": [_f(t) for t in fold_t],
            "fit_grade_medians": f_medians, "monotonic": fold_valid,
        })
        for idx, value in evaluated["value"].items():
            cv_grade[idx] = g.grade(value, fold_t)
            cv_score[idx] = g.score(value, fold_t)
            cv_basis[idx] = name
        for stage in sorted(evaluated["damage_stage"].unique()):
            stage_values = evaluated.loc[evaluated["damage_stage"] == stage, "value"]
            stage_median = g.median(stage_values)
            expected = spec.stage_grade[stage]
            got = g.grade(stage_median, fold_t) if stage_median is not None else None
            generalization.append({
                "fold": name, "stage": stage, "n": int(len(stage_values)),
                "median": _f(stage_median), "expected": expected, "got": got,
                "ok": got == expected,
            })

    # condition 1, 2
    c1 = g.exceed_rate(normal["value"], t1)
    danger = damage[damage["damage_stage"].isin(spec.danger_stages)]
    c2 = g.exceed_rate(danger["value"], t1)
    c2_by_stage = {
        stage: g.exceed_rate(danger.loc[danger["damage_stage"] == stage, "value"], t1)
        for stage in spec.danger_stages
    }
    danger_grades = cv_grade.reindex(danger.index)
    danger_as_danger = (
        float((danger_grades == "위험").mean()) if len(danger_grades) else None
    )

    # condition 3
    order_medians = {"정상": g.median(normal["value"])}
    order_medians.update(grade_medians)
    ordered = [order_medians[k] for k in g.GRADES]
    order_ok = all(v is not None for v in ordered) and all(
        a < b for a, b in zip(ordered, ordered[1:])
    )
    monotonic_all = g.thresholds_valid(final) and all(f["monotonic"] for f in fold_results)
    stage_medians = {
        stage: g.median(damage.loc[damage["damage_stage"] == stage, "value"])
        for stage in spec.stage_grade
    }
    stage_medians[f"{spec.normal_stage} test"] = order_medians["정상"]

    # condition 4
    extent = []
    for greater, lesser in spec.extent_pairs:
        a, b = stage_medians.get(greater), stage_medians.get(lesser)
        extent.append({
            "pair": f"{greater} > {lesser}", "ok": a is not None and b is not None and a > b,
        })

    conditions = {
        "1_false_alarm": {
            **c1, "limit": FA_LIMIT,
            "pass": c1["rate"] is not None and c1["rate"] <= FA_LIMIT,
        },
        "2_danger_detection": {
            **c2, "by_stage": c2_by_stage, "min": DETECTION_MIN,
            "graded_danger_ratio": danger_as_danger,
            "pass": c2["rate"] is not None and c2["rate"] >= DETECTION_MIN,
        },
        "3_grade_order": {
            "grade_medians": order_medians, "thresholds_monotonic": monotonic_all,
            "pass": bool(order_ok and monotonic_all),
        },
        "5_generalization": {
            "checks": generalization,
            "pass": bool(generalization) and all(c["ok"] for c in generalization),
        },
    }
    if spec.extent_pairs:
        conditions["4_extent_order"] = {
            "checks": extent, "pass": all(c["ok"] for c in extent),
        }

    # counts and provisional marks
    counts = {"ai2_calib": _counts(frame, frame["ai1_split"] == "ai2_calib")}
    counts[f"{spec.normal_stage} test"] = _counts(
        frame,
        (frame["ai1_split"] == "test") & (frame["label_status"] == MAIN_LABEL)
        & (frame["damage_stage"] == spec.normal_stage),
    )
    for stage in spec.stage_grade:
        counts[stage] = _counts(
            frame,
            (frame["ai1_split"] == "test") & (frame["label_status"] == MAIN_LABEL)
            & (frame["damage_stage"] == stage),
        )

    # reference rows (label 참고 etc.) graded with final thresholds
    reference = {}
    ref_rows = valid[(valid["ai1_split"] == "test") & (valid["label_status"] != MAIN_LABEL)]
    for (stage, label), part in ref_rows.groupby(["damage_stage", "label_status"]):
        grades = part["value"].apply(lambda v: g.grade(v, final))
        reference[f"{stage} ({label})"] = {
            "n": int(len(part)), "median": _f(g.median(part["value"])),
            "exceed_t1": g.exceed_rate(part["value"], t1),
            "grades": grades.value_counts(dropna=False).to_dict(),
        }

    # per measurement medians
    per_block = []
    block_rows = valid[valid["ai1_split"].isin(["ai2_calib", "test"])]
    for (exp, stage, split, label), part in block_rows.groupby(
        ["experiment_id", "damage_stage", "ai1_split", "label_status"]
    ):
        per_block.append({
            "experiment_id": exp, "stage": stage, "split": split, "label": label,
            "n": int(len(part)), "median": _f(g.median(part["value"])),
            "exceed_t1": g.exceed_rate(part["value"], t1)["rate"],
        })

    summary = {
        "thresholds_final": dict(zip(["T0", "T1", "T2", "T3", "T4"], [_f(t) for t in final])),
        "t1_uncertainty": g.t1_uncertainty(calib["value"]),
        "folds": fold_results,
        "stage_medians": {k: _f(v) for k, v in stage_medians.items()},
        "conditions": conditions,
        "counts": counts,
        "reference": reference,
        "per_block": per_block,
    }

    windows = frame[["experiment_id", "window_start_us", "ai1_split", "damage_stage",
                     "label_status", "cv_group", "held", "valid", "value"]].copy()
    windows["grade"] = None
    windows["score"] = np.nan
    windows["grade_basis"] = None
    for idx in windows.index[windows["valid"]]:
        if idx in cv_grade.index:
            windows.at[idx, "grade"] = cv_grade[idx]
            windows.at[idx, "score"] = cv_score[idx]
            windows.at[idx, "grade_basis"] = f"held-out ({cv_basis[idx]})"
        else:
            value = windows.at[idx, "value"]
            windows.at[idx, "grade"] = g.grade(value, final)
            windows.at[idx, "score"] = g.score(value, final)
            windows.at[idx, "grade_basis"] = "final"
    windows.loc[windows["held"], "grade"] = "판정 보류"
    windows.loc[~windows["held"] & windows["value"].isna(), "grade"] = "판정 불가"
    return summary, windows


def verdict(conditions: dict) -> str:
    def ok(key):
        return conditions.get(key, {}).get("pass", True)
    if not ok("1_false_alarm") or not ok("2_danger_detection"):
        return "출력 불가 (조건 1 또는 2 실패: 기준 대비 이상 여부도 출력하지 않음)"
    if not ok("3_grade_order"):
        return "4단계 등급 출력 불가 (조건 3 실패)"
    if not ok("4_extent_order") or not ok("5_generalization"):
        return "4단계 등급 검증 미완료 (조건 4 또는 5 실패)"
    return "통과 (공개 데이터 기준. API 연결 안 함)"


# ---------------------------------------------------------------- Box extras

def box_work_effect(raw: pd.DataFrame) -> dict:
    """Prereg 7 / agreement 3.4 on raw tac_4_2_z."""
    frame = prepare(raw, raw["tac_4_2_z"], "held_main")
    calib = frame[frame["valid"] & (frame["ai1_split"] == "ai2_calib")]
    _, t1 = g.lower_thresholds(calib["value"])
    base = (frame["ai1_split"] == "test") & (frame["damage_stage"] == "DS2") & (
        frame["label_status"] == MAIN_LABEL)
    out = {"raw_T1": _f(t1)}
    for subset in ("작업전", "작업중"):
        part = frame[base & (frame["subset"] == subset)]
        held_out = g.exceed_rate(part.loc[part["valid"], "value"], t1)
        before_hold = g.exceed_rate(part["value"], t1)
        out[subset] = {"after_hold": held_out, "before_hold_reference": before_hold}
    pre = out["작업전"]["after_hold"]
    n = pre["n"]
    if n < 10:
        out["work_effect_limit"] = None
        out["note"] = "n < 10: 참고만"
    else:
        out["work_effect_limit"] = bool(pre["exceed"] is not None and pre["exceed"] > FA_LIMIT * n)
        out["note"] = "오경보 수 > 0.05 × n 이면 '정상 기준이 작업 진동에 의존' 한계 기록"
    return out


def box_session_difference(raw: pd.DataFrame) -> dict:
    """Prereg 7 / agreement 3.5 on raw tac_2_3_z."""
    frame = prepare(raw, raw["tac_2_3_z"], "held_session")
    valid = frame[frame["valid"]]
    evening = valid[(valid["ai1_split"] == "session") & (valid["subset"] == "저녁")]
    day = valid[(valid["ai1_split"] == "session") & (valid["subset"] == "낮")]
    m_evening, m_day = g.median(evening["value"]), g.median(day["value"])
    gap = abs(m_evening - m_day) if m_evening is not None and m_day is not None else None
    out = {
        "n_evening": int(len(evening)), "n_day": int(len(day)),
        "median_evening": _f(m_evening), "median_day": _f(m_day), "gap": _f(gap),
        "reference_only": len(evening) < MIN_WINDOWS or len(day) < MIN_WINDOWS,
        "stages": {},
        "display": "세션 차이: 원본으로 확인, 열화본 미검증",
    }
    for stage in ("DS3", "DS5", "DS7"):
        part = valid[(valid["ai1_split"] == "test") & (valid["label_status"] == MAIN_LABEL)
                     & (valid["damage_stage"] == stage)]
        m = g.median(part["value"])
        shift = (m - m_day) if m is not None and m_day is not None else None
        out["stages"][stage] = {
            "n": int(len(part)), "median": _f(m), "shift_from_day": _f(shift),
            "provisional": None if shift is None or gap is None else bool(shift <= gap),
        }
    return out


# ---------------------------------------------------------------- HBTA extras

def hbta_values(raw: pd.DataFrame, deg: pd.DataFrame) -> dict:
    """AR distances from each kind's own AI1 train block (held excluded)."""
    out = {}
    for kind, frame, axes in (("deg_z", deg, "z"), ("raw_z", raw, "z"), ("raw_yz", raw, "yz")):
        reference = frame[(frame["ai1_split"] == "train") & ~frame["held_main"]]
        values, info = g.mahalanobis(reference, frame, g.ar_columns(axes))
        out[kind] = (frame, values, info)
    return out


def hbta_cause(deg_conditions: dict, raw_z: dict, raw_yz: dict) -> dict:
    causes = {}
    for key, result in deg_conditions.items():
        if result.get("pass", True):
            continue
        if raw_z.get(key, {}).get("pass"):
            causes[key] = "센서 잡음 (열화본 z 실패, 원본 z 통과)"
        elif raw_yz.get(key, {}).get("pass"):
            causes[key] = "y축 손실 (원본 z 실패, 원본 y+z 통과)"
        else:
            causes[key] = "지표 또는 데이터 한계 (원본 y+z도 실패)"
    return causes


def hbta_session(frame: pd.DataFrame) -> dict:
    valid = frame[frame["valid"] & (frame["damage_stage"] == "UDS")
                  & frame["ai1_split"].isin(["ai2_calib", "test"])]
    session = valid["source_record"].astype(str).str.extract(r"_(\d+)$")[0]
    m1 = g.median(valid.loc[session == "01", "value"])
    m2 = g.median(valid.loc[session == "02", "value"])
    return {
        "UDS_01": {"n": int((session == "01").sum()), "median": _f(m1)},
        "UDS_02": {"n": int((session == "02").sum()), "median": _f(m2)},
        "gap": _f(abs(m1 - m2)) if m1 is not None and m2 is not None else None,
    }


def hbta_uds01_reference(frame: pd.DataFrame) -> dict:
    valid = frame[frame["valid"] & (frame["damage_stage"] == "UDS")]
    uds01 = valid["source_record"].astype(str).str.endswith("_01")
    calib = valid[uds01 & (valid["ai1_split"] == "ai2_calib")]
    test = valid[uds01 & (valid["ai1_split"] == "test")]
    _, t1 = g.lower_thresholds(calib["value"])
    return {"T1": _f(t1), "n_calib": int(len(calib)), "false_alarm": g.exceed_rate(test["value"], t1)}


def hbta_conditions_table(frame: pd.DataFrame) -> list:
    rows = []
    for stage, part in frame[frame["ai1_split"].isin(["ai2_calib", "test"])].groupby("damage_stage"):
        rows.append({
            "stage": stage,
            "temperature_c": sorted(pd.to_numeric(part["temperature_c"], errors="coerce").dropna().unique().tolist()),
            "train_passing": sorted(part["train_passing"].astype(str).unique().tolist()),
        })
    return rows


# ---------------------------------------------------------------- Old ADA

def old_ada_report(frame: pd.DataFrame, kind: str) -> dict:
    reference = frame[(frame["ai1_split"] == "train") & ~frame["held_main"]]
    ar, info = g.standardized_euclid(reference, frame, g.ar_columns("z"))
    indicators = {"ar_distance": ar, "tac_a_b_z": pd.to_numeric(frame["tac_a_b_z"], errors="coerce")}
    out = {"reference_info": info, "indicators": {}}
    if kind == "deg":
        out["unit_note"] = "단위 m/s² 간접 추정, gal이면 열화본 전부 판정 보류"
    for name, values in indicators.items():
        data = prepare(frame, values, "held_main")
        valid = data[data["valid"]]
        ref_line = valid.loc[valid["ai1_split"] == "ai2_calib", "value"]
        line = float(ref_line.max()) if len(ref_line) else None
        groups = {}
        for (stage, label, split), part in valid.groupby(["damage_stage", "label_status", "ai1_split"]):
            groups[f"{stage} | {label} | {split}"] = {
                "n": int(len(part)), "median": _f(g.median(part["value"])),
                "above_reference_line": int((part["value"] > line).sum()) if line is not None else None,
            }
        normal = valid[(valid["damage_stage"] == "INT") & (valid["ai1_split"] == "test")
                       & (valid["label_status"] == MAIN_LABEL)]
        order_keys = [("INT", MAIN_LABEL), ("DMG1", MAIN_LABEL), ("DMG2", MAIN_LABEL), ("DMG3", MAIN_LABEL)]
        order = []
        for stage, label in order_keys:
            part = valid[(valid["damage_stage"] == stage) & (valid["label_status"] == label)
                         & (valid["ai1_split"] == "test")]
            order.append((stage, g.median(part["value"])))
        ranked = sorted([o for o in order if o[1] is not None], key=lambda o: o[1])
        run_columns = ["experiment_id", "damage_stage", "label_status", "ai1_split", "speed_kmh", "held", "value"]
        runs = data[[c for c in run_columns if c in data]]
        out["indicators"][name] = {
            "reference_line_ai2_calib_max": _f(line),
            "normal_test_above_line": int((normal["value"] > line).sum()) if line is not None else None,
            "normal_test_n": int(len(normal)),
            "groups": groups,
            "test_median_order_low_to_high": [r[0] for r in ranked],
            "runs": runs.assign(value=runs["value"].astype(float)).to_dict("records"),
        }
    return out
