"""XGBoost development experiment on HBTA sweep (SM) data: raw vs degraded.

DEVELOPMENT ONLY. All HBTA data has already been seen (v1, v2, grade v3), so
every number here is "결과 열람 후 실험" and is not performance evidence. No
model artifact is saved and nothing is connected to the API; the production
training pipeline (src/train.py, provenance gate) is not used or loosened.

Design (fixed before the first run, 2026-11-19):
  data      HBTA SM: AI1-public-1.2 (34 recordings) + 1.2-dev (6 normal)
            4 conditions (P1 Y, P1 Z, P2 Y, P2 Z) x (2 normal + DS1-DS8)
  windows   degraded hold_main = False (same windows for raw and deg models)
  features  window-intrinsic AI① features only (FEATURES below); no 1-TAC,
            no AI① v2 features (they depend on AI① NM references), no AI①
            results, no labels / dates / temperature / train passing
  models    two separate XGBoost models: raw features, degraded features
  tasks     binary: normal (UDS) vs damaged (DS1-DS8)
            grade : 정상 / 주의 / 경고 / 위험 by the v1 mapping
  split     leave one condition out (4 folds): train on 3 conditions,
            evaluate on the held-out condition (recordings never seen in training)
  params    fixed, no tuning (PARAMS), class weights balanced per fold
  metrics   AUC (normal vs damaged, normal vs danger DS2+DS8), at p > 0.5:
            false alarm on normal windows, detection on danger and all damage,
            per-state detection; grade accuracy and per-state grade mix;
            top features by gain

Usage: python -m src.public_grade.xgb_sm_experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from src.public_grade import truss_v2 as tv
from src.public_grade.grade_v3 import AXIS, sensor_labels

SEED = 20261109
CONDITIONS = ("SM_P1_Y", "SM_P1_Z", "SM_P2_Y", "SM_P2_Z")
DANGER = ("DS2", "DS8")
GRADE_OF = {"UDS": "정상", "DS3": "주의", "DS5": "주의", "DS6": "주의",
            "DS1": "경고", "DS4": "경고", "DS7": "경고", "DS2": "위험", "DS8": "위험"}
GRADES = ("정상", "주의", "경고", "위험")

STATS = ("mean", "rms", "std", "max", "min", "p2p", "crest", "dom_freq",
         "band_2_8", "band_8_15", "band_15_25")
FEATURES = (
    [f"{s}_{k}_{a}" for s in ("acc01", "acc02") for a in ("y", "z") for k in STATS]
    + ["cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2"]
    + [f"{s}_ar{k}_{a}" for s in ("acc01", "acc02") for a in ("y", "z") for k in range(1, 6)]
    + ["window_start_s"]
)
PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
              tree_method="hist", random_state=SEED, n_jobs=4)


def load_frames() -> dict:
    check = tv.load(Path("data/raw/ai1_public_1.2"), "hbta_as.csv")
    dev = tv.load(Path("data/raw/ai1_public_1.2_dev"), "hbta_as_dev.csv")
    out = {}
    for kind in ("raw", "deg"):
        frame = pd.concat([getattr(check, kind), getattr(dev, kind)], ignore_index=True)
        frame = frame[~frame["held"]].copy()
        frame["damaged"] = (frame["damage_stage"] != "UDS").astype(int)
        frame["grade"] = frame["damage_stage"].map(GRADE_OF)
        out[kind] = frame.reset_index(drop=True)
    keys = ["experiment_id", "window_start_us"]
    if not out["raw"][keys].equals(out["deg"][keys]):
        raise ValueError("raw and deg windows differ")
    return out


def auc(neg, pos) -> float | None:
    a, b = np.asarray(neg, float), np.asarray(pos, float)
    if not len(a) or not len(b):
        return None
    return float((b[:, None] > a[None, :]).mean() + 0.5 * (b[:, None] == a[None, :]).mean())


STAT_NAME = {"mean": "평균", "rms": "RMS", "std": "표준편차", "max": "최댓값", "min": "최솟값",
             "p2p": "최대-최소 폭", "crest": "파고율", "dom_freq": "주된 진동수(Hz)",
             "band_2_8": "2–8Hz 에너지 비율", "band_8_15": "8–15Hz 에너지 비율", "band_15_25": "15–25Hz 에너지 비율"}
TOP_N = 3


def feature_labels(labels: dict) -> dict:
    """Readable name with sensor and location for every model input."""
    out = {}
    for f in FEATURES:
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


ID_COLUMNS = ["experiment_id", "window_start_us", "source_record", "condition", "damage_stage"]
LABEL_COLUMNS = ["damaged", "grade"]
MODEL_STATUS = "개발용 실험 모델 (이미 본 데이터). 배포·API 연결 금지, models/measured/ 아님"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tv._jsonable(obj), ensure_ascii=False, indent=1))


def save_dataset(frame: pd.DataFrame, path: Path) -> None:
    """Exact model input: id columns, labels (not inputs), then the 67 input features."""
    out = frame[ID_COLUMNS + LABEL_COLUMNS + FEATURES]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def fold_plan(frame: pd.DataFrame) -> list[dict]:
    plan = []
    for held in CONDITIONS:
        test = frame["condition"] == held
        plan.append({
            "model_folder": f"heldout_{held}",
            "train_conditions": [c for c in CONDITIONS if c != held], "test_condition": held,
            "n_train_windows": int((~test).sum()), "n_test_windows": int(test.sum()),
            "train_records": sorted(frame.loc[~test, "source_record"].unique()),
            "test_records": sorted(frame.loc[test, "source_record"].unique()),
        })
    return plan


def binary_metrics(part: pd.DataFrame) -> dict:
    normal, dmg = part[part["damage_stage"] == "UDS"], part[part["damage_stage"] != "UDS"]
    danger = part[part["damage_stage"].isin(DANGER)]
    return {
        "auc_normal_vs_damaged": auc(normal["p_damaged"], dmg["p_damaged"]),
        "auc_normal_vs_danger": auc(normal["p_damaged"], danger["p_damaged"]),
        "false_alarm": float(normal["pred_damaged"].mean()),
        "false_alarm_by_record": normal.groupby("source_record")["pred_damaged"].mean().round(3).to_dict(),
        "detection_danger": float(danger["pred_damaged"].mean()),
        "detection_all_damage": float(dmg["pred_damaged"].mean()),
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


def run_kind(frame: pd.DataFrame, names: dict, out: Path, kind: str) -> tuple[list, pd.DataFrame, pd.Series]:
    """Train and evaluate the 8 models of one data kind; write each model's folder."""
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    name_list = [names[f] for f in FEATURES]
    codes = frame["grade"].map({g: i for i, g in enumerate(GRADES)})
    keep = ID_COLUMNS + ["grade", "window_start_s"]
    folds, preds, gains = [], [], []
    for plan in fold_plan(frame):
        held = plan["test_condition"]
        test = frame["condition"] == held
        train = ~test
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
                  "dataset_file": f"dataset/{kind}_dataset.csv", "features": len(FEATURES), "params": PARAMS}
        for task, model, part, metrics in (("binary", binary, b_part, b_metrics), ("grade", multi, g_part, g_metrics)):
            folder = out / kind / task / plan["model_folder"]
            folder.mkdir(parents=True, exist_ok=True)
            model.save_model(folder / "model.json")
            part.to_csv(folder / "predictions.csv", index=False, encoding="utf-8-sig")
            write_json(folder / "metrics.json", {**common, "task": task, "metrics": metrics})

        merged = b_part.join(g_part[["pred_grade", "grade_reason"]])
        preds.append(merged)
        gains.append(pd.Series(binary.get_booster().get_score(importance_type="gain")))
        folds.append({"held_out": held, "n_train": plan["n_train_windows"], "n_test": plan["n_test_windows"],
                      **b_metrics, "grade_accuracy": g_metrics["grade_accuracy"]})
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    return folds, pd.concat(preds), gain


def performance_table(out: Path) -> pd.DataFrame:
    """One row per model (16): data kind, task, held-out condition and its metrics."""
    rows = []
    for f in sorted(out.glob("*/*/heldout_*/metrics.json")):
        m = json.loads(f.read_text())
        x = m["metrics"]
        row = {"데이터": m["data_kind"], "과제": {"binary": "정상vs손상", "grade": "4단계 등급"}[m["task"]],
               "평가 조건": m["test_condition"], "학습 조건": "+".join(m["train_conditions"]),
               "학습 Window": m["n_train_windows"], "평가 Window": m["n_test_windows"]}
        if m["task"] == "binary":
            row.update({"AUC 정상vs손상": round(x["auc_normal_vs_damaged"], 3),
                        "AUC 정상vs위험": round(x["auc_normal_vs_danger"], 3),
                        "오경보율": round(x["false_alarm"], 3),
                        "위험 탐지율": round(x["detection_danger"], 3),
                        "손상 전체 탐지율": round(x["detection_all_damage"], 3)})
        else:
            row["등급 정확도"] = round(x["grade_accuracy"], 3)
        row["모델 폴더"] = str(f.parent.relative_to(out))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["과제", "데이터", "평가 조건"], ascending=[False, False, True])
    means = (table.groupby(["데이터", "과제"], sort=False).mean(numeric_only=True).round(3).reset_index())
    means["평가 조건"] = "평균 (4개)"
    table = pd.concat([table, means], ignore_index=True)
    table.to_csv(out / "performance.csv", index=False, encoding="utf-8-sig")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/public_grade/xgb_sm"))
    parser.add_argument("--summary", type=Path, default=Path("results/public_grade/xgb_sm_summary.json"))
    args = parser.parse_args(argv)
    frames = load_frames()
    names = feature_labels(sensor_labels(Path("data/raw/ai1_public_1.2"), "hbta"))
    summary = {"note": "개발용 실험 (이미 본 데이터, 결과 열람 후). 성능 근거 아님. 모델은 개발용 폴더에만 저장, API 연결 없음",
               "design": __doc__.split("Design")[1].split("Usage")[0].strip(),
               "features": FEATURES, "params": PARAMS, "code_commit": tv.git_commit(), "models": {}}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset" / "folds.json", {
        "note": "모델 폴더 heldout_<조건> = 그 조건을 빼고 학습, 그 조건으로 평가",
        "input_features": [{"column": f, "name": names[f]} for f in FEATURES],
        "label_columns": {"damaged": "0 정상(UDS), 1 손상(DS1–DS8)", "grade": "v1 기대 등급 (정상/주의/경고/위험)"},
        "folds": fold_plan(frames["deg"]),
    })
    for kind, frame in frames.items():
        save_dataset(frame, out / "dataset" / f"{kind}_dataset.csv")
        folds, preds, gain = run_kind(frame, names, out, kind)
        preds.to_csv(out / kind / "all_predictions.csv", index=False, encoding="utf-8-sig")
        mix = pd.crosstab(preds["damage_stage"], preds["pred_grade"], normalize="index").reindex(columns=GRADES).fillna(0)
        summary["models"][kind] = {
            "n_windows": int(len(frame)), "folds": folds,
            "mean": {k: float(np.mean([f[k] for f in folds])) for k in
                     ("auc_normal_vs_damaged", "auc_normal_vs_danger", "false_alarm",
                      "detection_danger", "detection_all_damage", "grade_accuracy")},
            "grade_mix_by_state": mix.round(3).to_dict(orient="index"),
            "top_features_gain": {f"{k} ({names[k]})": v for k, v in gain.head(15).round(2).items()},
        }
        m = summary["models"][kind]["mean"]
        print(f"[{kind}] AUC 정상vs손상 {m['auc_normal_vs_damaged']:.3f}, 정상vs위험 {m['auc_normal_vs_danger']:.3f}, "
              f"오경보 {m['false_alarm']:.3f}, 위험 탐지 {m['detection_danger']:.3f}, "
              f"손상 전체 탐지 {m['detection_all_damage']:.3f}, 등급 정확도 {m['grade_accuracy']:.3f}")
    write_json(args.summary, summary)
    performance_table(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
