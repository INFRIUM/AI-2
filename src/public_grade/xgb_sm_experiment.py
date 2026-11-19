"""XGBoost development experiment on HBTA sweep (SM) data: raw vs degraded.

DEVELOPMENT ONLY. All HBTA data has already been seen (v1, v2, grade v3), so
every number here is "결과 열람 후 실험" and is not performance evidence.
Shared training, outputs and run bookkeeping live in xgb_engine.py.

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

from pathlib import Path

import pandas as pd

from src.public_grade import truss_v2 as tv
from src.public_grade import xgb_engine as xe
from src.public_grade.xgb_engine import resolve_run, update_index  # noqa: F401 (used by tests)

CONDITIONS = ("SM_P1_Y", "SM_P1_Z", "SM_P2_Y", "SM_P2_Z")
FEATURES = xe.BASE_FEATURES + ["window_start_s"]
CHECK_DIR, DEV_DIR = Path("data/raw/ai1_public_1.2"), Path("data/raw/ai1_public_1.2_dev")
INPUT_FILES = [CHECK_DIR / "hbta_anomaly_results_raw.csv", CHECK_DIR / "hbta_anomaly_results_deg.csv",
               DEV_DIR / "hbta_anomaly_results_raw.csv", DEV_DIR / "hbta_anomaly_results_deg.csv"]
EVAL_DESIGN = "leave-one-condition-out v1, held = degraded hold_main, p > 0.5"


def load_frames() -> dict:
    check = tv.load(CHECK_DIR, "hbta_as.csv")
    dev = tv.load(DEV_DIR, "hbta_as_dev.csv")
    out = {kind: xe.label_frame(pd.concat([getattr(check, kind), getattr(dev, kind)], ignore_index=True))
           for kind in ("raw", "deg")}
    keys = ["experiment_id", "window_start_us"]
    if not out["raw"][keys].equals(out["deg"][keys]):
        raise ValueError("raw and deg windows differ")
    return out


def fold_plan(frame: pd.DataFrame) -> list[dict]:
    plan = []
    for held in CONDITIONS:
        test = frame["condition"] == held
        others = [c for c in CONDITIONS if c != held]
        plan.append({
            "model_folder": f"heldout_{held}", "test_label": held, "train_label": "+".join(others),
            "n_train_windows": int((~test).sum()), "n_test_windows": int(test.sum()),
            "train_records": sorted(frame.loc[~test, "source_record"].unique()),
            "test_records": sorted(frame.loc[test, "source_record"].unique()),
        })
    return plan


SPEC = xe.Spec(
    name="hbta_sm", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN,
    design_doc=__doc__.split("Design")[1].split("Usage")[0].strip(),
    eval_label="조건 하나 빼기 (4개 조건)", load=load_frames, plan=fold_plan,
    sensor_dir=CHECK_DIR, extra_columns=("window_start_s",),
)


def main(argv=None) -> int:
    return xe.main(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
