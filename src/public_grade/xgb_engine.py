"""Shared engine for XGBoost development experiments on AI① public data.

A dataset module (xgb_sm_experiment, xgb_nm_experiment) supplies a Spec: input
files, features, how to load windows and how to split into folds. The engine
trains raw and degraded models for two tasks (normal vs damaged, 4-level
grade), writes every model's folder, a performance table and an accumulating
run index.

DEVELOPMENT ONLY: models are saved under data/processed/ for inspection, never
under models/measured/ and never connected to the API. The production training
pipeline (src/train.py, provenance gate) is not used or loosened.

Runs: a training run is identified by an input fingerprint (input file MD5s +
features + params + evaluation design). A new fingerprint creates a new run
<dataset>_<YYYYMMDD>_<fingerprint8>; the same fingerprint regenerates the
existing run instead of piling up duplicates.
  results/public_grade/xgb_performance/        <run>.csv, <run>_summary.json, xgb_performance_index.csv
  data/processed/public_grade/xgb_runs/<run>/  dataset/, raw/, deg/, performance.csv, run_info.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from src.public_grade import truss_v2 as tv
from src.public_grade.grade_v3 import AXIS, sensor_labels

SEED = 20261109
DANGER = ("DS2", "DS8")
GRADE_OF = {"UDS": "정상", "DS3": "주의", "DS5": "주의", "DS6": "주의",
            "DS1": "경고", "DS4": "경고", "DS7": "경고", "DS2": "위험", "DS8": "위험"}
GRADES = ("정상", "주의", "경고", "위험")
STATS = ("mean", "rms", "std", "max", "min", "p2p", "crest", "dom_freq",
         "band_2_8", "band_8_15", "band_15_25")
BASE_FEATURES = (
    [f"{s}_{k}_{a}" for s in ("acc01", "acc02") for a in ("y", "z") for k in STATS]
    + ["cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2"]
    + [f"{s}_ar{k}_{a}" for s in ("acc01", "acc02") for a in ("y", "z") for k in range(1, 6)]
)
PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
              tree_method="hist", random_state=SEED, n_jobs=4)
INDEX_NAME = "xgb_performance_index.csv"
ID_COLUMNS = ["experiment_id", "window_start_us", "source_record", "condition", "damage_stage"]
LABEL_COLUMNS = ["damaged", "grade"]
MODEL_STATUS = "개발용 실험 모델 (이미 본 데이터). 배포·API 연결 금지, models/measured/ 아님"
STAT_NAME = {"mean": "평균", "rms": "RMS", "std": "표준편차", "max": "최댓값", "min": "최솟값",
             "p2p": "최대-최소 폭", "crest": "파고율", "dom_freq": "주된 진동수(Hz)",
             "band_2_8": "2–8Hz 에너지 비율", "band_8_15": "8–15Hz 에너지 비율", "band_15_25": "15–25Hz 에너지 비율"}
TOP_N = 3


@dataclass
class Spec:
    name: str                                   # dataset tag in the run id, e.g. "hbta_sm"
    input_files: list                           # files whose MD5 enter the fingerprint
    features: list                              # model inputs
    design: str                                 # evaluation design string (enters the fingerprint)
    design_doc: str                             # human-readable design (summary only)
    eval_label: str                             # short description for the run index
    load: Callable[[], dict]                    # -> {"raw": frame, "deg": frame}
    plan: Callable[[pd.DataFrame], list]        # -> list of fold dicts (see fold dict keys below)
    sensor_dir: Path                            # folder with sensor_info.csv
    extra_columns: tuple = ()                   # extra columns kept in predictions


# fold dict keys: model_folder, test_label, train_label, train_records, test_records,
#                 n_train_windows, n_test_windows


# ------------------------------------------------------------------ runs

def input_fingerprint(spec: Spec) -> tuple[str, dict]:
    """Same input files + same features + same settings -> same fingerprint (no duplicate run)."""
    digests = {str(p): tv.loader.file_digest(p) for p in spec.input_files}
    blob = json.dumps({"inputs": digests, "features": list(spec.features), "params": PARAMS,
                       "design": spec.design}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode()).hexdigest(), digests


def resolve_run(index_path: Path, fingerprint: str, dataset: str, today: str) -> tuple[str, bool]:
    """Reuse the run id when the fingerprint is already in the index, else create a new one."""
    if index_path.exists():
        index = pd.read_csv(index_path, encoding="utf-8-sig", dtype=str)
        hit = index[index["입력 지문"] == fingerprint]
        if not hit.empty:
            return hit["학습 건"].iloc[0], False
    return f"{dataset}_{today}_{fingerprint[:8]}", True


def update_index(index_path: Path, row: dict) -> pd.DataFrame:
    index = pd.read_csv(index_path, encoding="utf-8-sig", dtype=str) if index_path.exists() else pd.DataFrame()
    if not index.empty:
        index = index[index["학습 건"] != row["학습 건"]]
    index = pd.concat([index, pd.DataFrame([row]).astype(str)], ignore_index=True).sort_values("학습 건")
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    return index


# ------------------------------------------------------------------ helpers

def label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[~frame["held"]].copy()
    frame["damaged"] = (frame["damage_stage"] != "UDS").astype(int)
    frame["grade"] = frame["damage_stage"].map(GRADE_OF)
    return frame.reset_index(drop=True)


def auc(neg, pos) -> float | None:
    a, b = np.asarray(neg, float), np.asarray(pos, float)
    if not len(a) or not len(b):
        return None
    return float((b[:, None] > a[None, :]).mean() + 0.5 * (b[:, None] == a[None, :]).mean())


def feature_labels(features: list, labels: dict) -> dict:
    """Readable name with sensor and location for every model input."""
    out = {}
    for f in features:
        parts = f.split("_")
        if f == "window_start_s":
            out[f] = "스윕 진행 시각(초, 가진 주파수 위치)"
        elif f == "cross_amp_ratio_L4_L2_z":
            out[f] = f"{labels['acc01']}/{labels['acc02']} 수직 진폭비"
        elif f == "cross_torsion_y_z_L2":
            out[f] = f"{labels['acc02']} 횡방향/수직 비(비틀림)"
        elif parts[1].startswith("ar"):
            out[f] = f"{labels[parts[0]]} {AXIS[parts[2]]} AR 계수 {parts[1][2:]}"
        else:
            stat = "_".join(parts[1:-1])
            out[f] = f"{labels[parts[0]]} {AXIS[parts[-1]]} {STAT_NAME[stat]}"
    return out


def _contrib_text(names, values, contribs, idx) -> str:
    return ", ".join(f"{names[i]} {values[i]:.3g} (영향 {contribs[i]:+.2f})" if np.isfinite(values[i])
                     else f"{names[i]} 값 없음 (영향 {contribs[i]:+.2f})" for i in idx)


def binary_reason(p: float, values: np.ndarray, contribs: np.ndarray, names: list) -> str:
    """판정 사유: probability vs 0.5 and the features that pushed most toward each side.
    contribs are SHAP values in log-odds (positive = toward 손상)."""
    decision = "손상" if p > 0.5 else "정상"
    head = f"손상 확률 {p:.2f} (기준 0.5 {'초과' if p > 0.5 else '이하'} → {decision})"
    order = np.argsort(contribs)
    toward_damage = [i for i in order[::-1] if contribs[i] > 0][:TOP_N]
    toward_normal = [i for i in order if contribs[i] < 0][:TOP_N]
    first, second = (toward_damage, toward_normal) if decision == "손상" else (toward_normal, toward_damage)
    first_name, second_name = ("손상", "정상") if decision == "손상" else ("정상", "손상")
    text = head
    if first:
        text += f" / {first_name} 쪽 근거: " + _contrib_text(names, values, contribs, first)
    if second:
        text += f" / 반대({second_name}) 쪽 근거: " + _contrib_text(names, values, contribs, second[:1])
    return text


def grade_reason(grade: str, prob: float, values: np.ndarray, contribs: np.ndarray, names: list) -> str:
    idx = [i for i in np.argsort(contribs)[::-1] if contribs[i] > 0][:TOP_N]
    text = f"예측 등급 {grade} (확률 {prob:.2f})"
    if idx:
        text += f" / {grade} 쪽 근거: " + _contrib_text(names, values, contribs, idx)
    return text


def balanced_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map(lambda v: len(y) / (len(counts) * counts[v])).to_numpy(float)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tv._jsonable(obj), ensure_ascii=False, indent=1))


def save_dataset(frame: pd.DataFrame, features: list, path: Path) -> None:
    """Exact model input: id columns, labels (not inputs), then the input features."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[ID_COLUMNS + LABEL_COLUMNS + list(features)].to_csv(path, index=False, encoding="utf-8-sig")


def binary_metrics(part: pd.DataFrame) -> dict:
    normal, dmg = part[part["damage_stage"] == "UDS"], part[part["damage_stage"] != "UDS"]
    danger = part[part["damage_stage"].isin(DANGER)]
    mean = lambda s: float(s.mean()) if len(s) else None
    return {
        "auc_normal_vs_damaged": auc(normal["p_damaged"], dmg["p_damaged"]),
        "auc_normal_vs_danger": auc(normal["p_damaged"], danger["p_damaged"]),
        "false_alarm": mean(normal["pred_damaged"]),
        "false_alarm_by_record": normal.groupby("source_record")["pred_damaged"].mean().round(3).to_dict(),
        "detection_danger": mean(danger["pred_damaged"]),
        "detection_all_damage": mean(dmg["pred_damaged"]),
        "detection_by_state": dmg.groupby("damage_stage")["pred_damaged"].mean().round(3).to_dict(),
    }


def grade_metrics(part: pd.DataFrame) -> dict:
    confusion = pd.crosstab(part["grade"], part["pred_grade"]).reindex(index=GRADES, columns=GRADES).fillna(0)
    mix = pd.crosstab(part["damage_stage"], part["pred_grade"], normalize="index").reindex(columns=GRADES).fillna(0)
    return {
        "grade_accuracy": float((part["pred_grade"] == part["grade"]).mean()),
        "confusion_true_by_pred": confusion.astype(int).to_dict(orient="index"),
        "pred_grade_mix_by_state": mix.round(3).to_dict(orient="index"),
    }


def _nanmean(values) -> float | None:
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------ training

def run_kind(spec: Spec, frame: pd.DataFrame, names: dict, out: Path, kind: str, plans: list):
    """Train and evaluate the models of one data kind; write each model's folder."""
    features = list(spec.features)
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    name_list = [names[f] for f in features]
    codes = frame["grade"].map({g: i for i, g in enumerate(GRADES)})
    keep = ID_COLUMNS + ["grade"] + list(spec.extra_columns)
    folds, preds, gains = [], [], []
    for plan in plans:
        test = frame["source_record"].isin(plan["test_records"])
        train = frame["source_record"].isin(plan["train_records"])
        dm = DMatrix(X[test])
        xv = X[test].to_numpy(float)

        binary = XGBClassifier(objective="binary:logistic", **PARAMS)
        binary.fit(X[train], frame.loc[train, "damaged"], sample_weight=balanced_weights(frame.loc[train, "damaged"]))
        p = binary.predict_proba(X[test])[:, 1]
        c_bin = binary.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        b_part = frame.loc[test, keep].copy()
        b_part["p_damaged"], b_part["pred_damaged"] = p, (p > 0.5).astype(int)
        b_part["reason"] = [binary_reason(p[i], xv[i], c_bin[i], name_list) for i in range(len(b_part))]

        multi = XGBClassifier(objective="multi:softprob", num_class=4, **PARAMS)
        multi.fit(X[train], codes[train], sample_weight=balanced_weights(codes[train]))
        grade_prob = multi.predict_proba(X[test])
        grade_idx = grade_prob.argmax(axis=1)
        grade_pred = np.array(GRADES)[grade_idx]
        c_multi = multi.get_booster().predict(dm, pred_contribs=True)[:, :, :-1]
        g_part = frame.loc[test, keep].copy()
        g_part["pred_grade"] = grade_pred
        for k, gname in enumerate(GRADES):
            g_part[f"p_{gname}"] = grade_prob[:, k]
        g_part["grade_reason"] = [grade_reason(grade_pred[i], grade_prob[i, grade_idx[i]], xv[i],
                                               c_multi[i, grade_idx[i]], name_list) for i in range(len(g_part))]

        b_metrics, g_metrics = binary_metrics(b_part), grade_metrics(g_part)
        common = {"status": MODEL_STATUS, "data_kind": kind, **plan,
                  "dataset_file": f"dataset/{kind}_dataset.csv", "features": len(features), "params": PARAMS}
        for task, model, part, metrics in (("binary", binary, b_part, b_metrics), ("grade", multi, g_part, g_metrics)):
            folder = out / kind / task / plan["model_folder"]
            folder.mkdir(parents=True, exist_ok=True)
            model.save_model(folder / "model.json")
            part.to_csv(folder / "predictions.csv", index=False, encoding="utf-8-sig")
            write_json(folder / "metrics.json", {**common, "task": task, "metrics": metrics})

        merged = b_part.join(g_part[["pred_grade", "grade_reason"]])
        merged.insert(0, "평가 모델", plan["model_folder"])
        preds.append(merged)
        gains.append(pd.Series(binary.get_booster().get_score(importance_type="gain")))
        folds.append({"held_out": plan["test_label"], "n_train": plan["n_train_windows"],
                      "n_test": plan["n_test_windows"], **b_metrics, "grade_accuracy": g_metrics["grade_accuracy"]})
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    # a normal recording can be evaluated by several models (NM), so renumber rows
    return folds, pd.concat(preds, ignore_index=True), gain


def performance_table(out: Path) -> pd.DataFrame:
    """One row per model: data kind, task, held-out set and its metrics, plus mean rows."""
    rows = []
    for f in sorted(out.glob("*/*/heldout_*/metrics.json")):
        m = json.loads(f.read_text())
        x = m["metrics"]
        row = {"데이터": m["data_kind"], "과제": {"binary": "정상vs손상", "grade": "4단계 등급"}[m["task"]],
               "평가 대상": m["test_label"], "학습 대상": m["train_label"],
               "학습 Window": m["n_train_windows"], "평가 Window": m["n_test_windows"]}
        if m["task"] == "binary":
            r3 = lambda v: None if v is None else round(v, 3)
            row.update({"AUC 정상vs손상": r3(x["auc_normal_vs_damaged"]), "AUC 정상vs위험": r3(x["auc_normal_vs_danger"]),
                        "오경보율": r3(x["false_alarm"]), "위험 탐지율": r3(x["detection_danger"]),
                        "손상 전체 탐지율": r3(x["detection_all_damage"])})
        else:
            row["등급 정확도"] = round(x["grade_accuracy"], 3)
        row["모델 폴더"] = str(f.parent.relative_to(out))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["과제", "데이터", "평가 대상"], ascending=[False, False, True])
    n = table.groupby(["데이터", "과제"]).size().iloc[0]
    means = table.groupby(["데이터", "과제"], sort=False).mean(numeric_only=True).round(3).reset_index()
    means["평가 대상"] = f"평균 ({n}개)"
    table = pd.concat([table, means], ignore_index=True)
    table.to_csv(out / "performance.csv", index=False, encoding="utf-8-sig")
    return table


METRIC_KEYS = ("auc_normal_vs_damaged", "auc_normal_vs_danger", "false_alarm",
               "detection_danger", "detection_all_damage", "grade_accuracy")


def main(spec: Spec, argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("data/processed/public_grade/xgb_runs"),
                        help="학습 건별 모델·예측·데이터셋 폴더 (git 제외)")
    parser.add_argument("--perf-dir", type=Path, default=Path("results/public_grade/xgb_performance"),
                        help="학습 건별 성능표와 학습 건 목록 (git 포함)")
    args = parser.parse_args(argv)
    fingerprint, digests = input_fingerprint(spec)
    index_path = args.perf_dir / INDEX_NAME
    run_id, is_new = resolve_run(index_path, fingerprint, spec.name, date.today().strftime("%Y%m%d"))
    print(f"학습 건: {run_id} ({'새 학습 건' if is_new else '같은 데이터·설정의 기존 학습 건을 다시 만듦 (쌓지 않음)'})")
    frames = spec.load()
    names = feature_labels(list(spec.features), sensor_labels(spec.sensor_dir, "hbta"))
    plans = spec.plan(frames["deg"])
    summary = {"run_id": run_id, "dataset": spec.name, "input_fingerprint": fingerprint, "inputs_md5": digests,
               "note": "개발용 실험 (이미 본 데이터, 결과 열람 후). 성능 근거 아님. 모델은 개발용 폴더에만 저장, API 연결 없음",
               "design": spec.design_doc, "features": list(spec.features), "params": PARAMS,
               "code_commit": tv.git_commit(), "models": {}}
    out = args.runs_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset" / "folds.json", {
        "note": "모델 폴더 heldout_<평가 대상> = 그 대상을 빼고 학습, 그 대상으로 평가",
        "input_features": [{"column": f, "name": names[f]} for f in spec.features],
        "label_columns": {"damaged": "0 정상(UDS), 1 손상(DS1–DS8)", "grade": "v1 기대 등급 (정상/주의/경고/위험)"},
        "folds": plans,
    })
    for kind, frame in frames.items():
        save_dataset(frame, spec.features, out / "dataset" / f"{kind}_dataset.csv")
        folds, preds, gain = run_kind(spec, frame, names, out, kind, plans)
        preds.to_csv(out / kind / "all_predictions.csv", index=False, encoding="utf-8-sig")
        mix = pd.crosstab(preds["damage_stage"], preds["pred_grade"], normalize="index").reindex(columns=GRADES).fillna(0)
        summary["models"][kind] = {
            "n_windows": int(len(frame)), "folds": folds,
            "mean": {k: _nanmean([f[k] for f in folds]) for k in METRIC_KEYS},
            "grade_mix_by_state": mix.round(3).to_dict(orient="index"),
            "top_features_gain": {f"{k} ({names[k]})": v for k, v in gain.head(15).round(2).items()},
        }
        m = summary["models"][kind]["mean"]
        print(f"[{kind}] AUC 정상vs손상 {m['auc_normal_vs_damaged']:.3f}, 정상vs위험 {m['auc_normal_vs_danger']:.3f}, "
              f"오경보 {m['false_alarm']:.3f}, 위험 탐지 {m['detection_danger']:.3f}, "
              f"손상 전체 탐지 {m['detection_all_damage']:.3f}, 등급 정확도 {m['grade_accuracy']:.3f}")
    table = performance_table(out)
    args.perf_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.perf_dir / f"{run_id}.csv", index=False, encoding="utf-8-sig")
    write_json(args.perf_dir / f"{run_id}_summary.json", summary)
    write_json(out / "run_info.json", {k: summary[k] for k in ("run_id", "dataset", "input_fingerprint",
                                                               "inputs_md5", "note", "params", "code_commit")})
    mean = {k: summary["models"][k]["mean"] for k in ("raw", "deg")}
    r3 = lambda v: "" if v is None else round(v, 3)
    index = update_index(index_path, {
        "학습 건": run_id, "학습일": run_id.split("_")[-2], "데이터셋": spec.name,
        "입력 지문": fingerprint, "Window 수": summary["models"]["deg"]["n_windows"],
        "모델 수": 4 * len(plans), "평가 방식": spec.eval_label,
        "원본 AUC": r3(mean["raw"]["auc_normal_vs_damaged"]), "열화본 AUC": r3(mean["deg"]["auc_normal_vs_damaged"]),
        "원본 오경보율": r3(mean["raw"]["false_alarm"]), "열화본 오경보율": r3(mean["deg"]["false_alarm"]),
        "원본 위험 탐지율": r3(mean["raw"]["detection_danger"]), "열화본 위험 탐지율": r3(mean["deg"]["detection_danger"]),
        "원본 등급 정확도": r3(mean["raw"]["grade_accuracy"]), "열화본 등급 정확도": r3(mean["deg"]["grade_accuracy"]),
        "성능표": f"{run_id}.csv", "모델 폴더": str(out), "코드 커밋": (summary["code_commit"] or "")[:7],
        "상태": "개발용 실험 (이미 본 데이터, 배포 금지)",
    })
    print(f"성능표: {args.perf_dir / (run_id + '.csv')}  |  학습 건 목록: {index_path} ({len(index)}건)")
    return 0
