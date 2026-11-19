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
from xgboost import XGBClassifier

from src.public_grade import truss_v2 as tv

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


def balanced_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map(lambda v: len(y) / (len(counts) * counts[v])).to_numpy(float)


def run_kind(frame: pd.DataFrame) -> tuple[list, pd.DataFrame, pd.Series]:
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    folds, preds, gains = [], [], []
    for held in CONDITIONS:
        test = frame["condition"] == held
        train = ~test
        binary = XGBClassifier(objective="binary:logistic", **PARAMS)
        binary.fit(X[train], frame.loc[train, "damaged"], sample_weight=balanced_weights(frame.loc[train, "damaged"]))
        p = binary.predict_proba(X[test])[:, 1]

        codes = frame["grade"].map({g: i for i, g in enumerate(GRADES)})
        multi = XGBClassifier(objective="multi:softprob", num_class=4, **PARAMS)
        multi.fit(X[train], codes[train], sample_weight=balanced_weights(codes[train]))
        grade_pred = np.array(GRADES)[multi.predict_proba(X[test]).argmax(axis=1)]

        part = frame.loc[test, ["condition", "source_record", "damage_stage", "grade", "window_start_s"]].copy()
        part["p_damaged"], part["pred_damaged"], part["pred_grade"] = p, (p > 0.5).astype(int), grade_pred
        preds.append(part)
        gains.append(pd.Series(binary.get_booster().get_score(importance_type="gain")))

        normal, dmg = part[part["damage_stage"] == "UDS"], part[part["damage_stage"] != "UDS"]
        danger = part[part["damage_stage"].isin(DANGER)]
        folds.append({
            "held_out": held,
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "auc_normal_vs_damaged": auc(normal["p_damaged"], dmg["p_damaged"]),
            "auc_normal_vs_danger": auc(normal["p_damaged"], danger["p_damaged"]),
            "false_alarm": float(normal["pred_damaged"].mean()),
            "false_alarm_by_record": normal.groupby("source_record")["pred_damaged"].mean().round(3).to_dict(),
            "detection_danger": float(danger["pred_damaged"].mean()),
            "detection_all_damage": float(dmg["pred_damaged"].mean()),
            "detection_by_state": dmg.groupby("damage_stage")["pred_damaged"].mean().round(3).to_dict(),
            "grade_accuracy": float((part["pred_grade"] == part["grade"]).mean()),
        })
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    return folds, pd.concat(preds), gain


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/public_grade/xgb_sm"))
    parser.add_argument("--summary", type=Path, default=Path("results/public_grade/xgb_sm_summary.json"))
    args = parser.parse_args(argv)
    frames = load_frames()
    summary = {"note": "개발용 실험 (이미 본 데이터, 결과 열람 후). 성능 근거 아님. 모델 저장·API 연결 없음",
               "design": __doc__.split("Design")[1].split("Usage")[0].strip(),
               "features": FEATURES, "params": PARAMS, "code_commit": tv.git_commit(), "models": {}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind, frame in frames.items():
        folds, preds, gain = run_kind(frame)
        preds.to_csv(args.out_dir / f"xgb_sm_{kind}_predictions.csv", index=False, encoding="utf-8-sig")
        mix = pd.crosstab(preds["damage_stage"], preds["pred_grade"], normalize="index").reindex(columns=GRADES).fillna(0)
        summary["models"][kind] = {
            "n_windows": int(len(frame)), "folds": folds,
            "mean": {k: float(np.mean([f[k] for f in folds])) for k in
                     ("auc_normal_vs_damaged", "auc_normal_vs_danger", "false_alarm",
                      "detection_danger", "detection_all_damage", "grade_accuracy")},
            "grade_mix_by_state": mix.round(3).to_dict(orient="index"),
            "top_features_gain": gain.head(15).round(2).to_dict(),
        }
        m = summary["models"][kind]["mean"]
        print(f"[{kind}] AUC 정상vs손상 {m['auc_normal_vs_damaged']:.3f}, 정상vs위험 {m['auc_normal_vs_danger']:.3f}, "
              f"오경보 {m['false_alarm']:.3f}, 위험 탐지 {m['detection_danger']:.3f}, "
              f"손상 전체 탐지 {m['detection_all_damage']:.3f}, 등급 정확도 {m['grade_accuracy']:.3f}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(tv._jsonable(summary), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
