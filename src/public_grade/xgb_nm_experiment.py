"""XGBoost development experiment on HBTA noise-mode (NM) data: raw vs degraded.

DEVELOPMENT ONLY. The 10 NM recordings were used in v1, exploration 5-1 and
grade v3, so every number is "결과 열람 후 실험", not performance evidence.
Shared training, outputs and run bookkeeping live in xgb_engine.py.

Design (fixed before the first run, 2026-11-19):
  data      HBTA NM (AI1-public-1.1): position P2, vertical excitation,
            UDS_01 (9/27), UDS_02 (9/22), DS1-DS8 one recording each (9/27-28)
  windows   degraded hold_main = False (no NM window is held)
  features  same window-intrinsic AI① features as the SM experiment, without
            window_start_s (NM has no sweep; time within a recording would only
            identify the recording). Degraded y-axis AR is almost all missing.
  split     a recording is never split. 8 folds, one per damage state: hold
            out that damage recording + one normal recording, train on the
            other 7 damage recordings + the other normal recording.
            Normal recording held out: UDS_02 (9/22) for DS1, DS3, DS5, DS7;
            UDS_01 (9/27) for DS2, DS4, DS6, DS8.
  tasks, params, metrics: same as the SM experiment (xgb_engine)
  limits    damage recordings are 9/27-28 only and one normal day is always
            missing from training, so a model can learn the date instead of
            damage; training normal data is a single day in every fold.

Variants (5-2, added 2026-11-20 before running them; --variant):
  base      the design above, decision p > 0.5 (run hbta_nm_20261119_1e3c3daa)
  p95       same models, decision threshold = 95th percentile of out-of-fold
            probabilities of the TRAINING normal windows (inner 5-fold split
            inside the training data; the normal recording is cut into 4 time
            blocks so it can appear in the inner split). Evaluated windows are
            never used to set the threshold.
  avg3_p95  features averaged over 3 consecutive windows of the same recording
            (causal; the first 2 windows of a recording are dropped), then p95

Usage: python -m src.public_grade.xgb_nm_experiment [--variant base|p95|avg3_p95]
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import loader
from src.public_grade import xgb_engine as xe

DATA_DIR = Path("data/raw/ai1_public_1.1")
FEATURES = list(xe.BASE_FEATURES)
INPUT_FILES = [DATA_DIR / "hbta_anomaly_results_raw.csv", DATA_DIR / "hbta_anomaly_results_deg.csv"]
EVAL_DESIGN = "leave-one-damage-recording-out with one normal recording (alternating day) v1, held = degraded hold_main, p > 0.5"
NORMAL = {"01": "MVS_P2_UDS_NM_Z_01", "02": "MVS_P2_UDS_NM_Z_02"}
NORMAL_DATE = {"01": "9/27", "02": "9/22"}


def average_windows(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mean of each feature over k consecutive windows ending at a window (same recording)."""
    frame = frame.sort_values(["source_record", "window_start_us"]).copy()
    rolled = frame.groupby("source_record")[FEATURES].rolling(k, min_periods=k).mean()
    frame[FEATURES] = rolled.reset_index(level=0, drop=True).loc[frame.index].to_numpy()
    first = frame.groupby("source_record").cumcount() < k - 1
    return frame[~first]


def load_frames(k: int = 1) -> dict:
    raw, deg = loader.load_pair(DATA_DIR, "hbta")
    held = deg["held_main"].to_numpy()
    out = {}
    for kind, frame in (("raw", raw), ("deg", deg)):
        frame = frame.copy()
        frame["held"] = held
        if k > 1:
            frame = average_windows(frame, k)
        out[kind] = xe.label_frame(frame)
    keys = ["experiment_id", "window_start_us"]
    if not out["raw"][keys].equals(out["deg"][keys]):
        raise ValueError("raw and deg windows differ")
    return out


def fold_plan(frame: pd.DataFrame) -> list[dict]:
    plan = []
    records = sorted(frame["source_record"].unique())
    for k in range(1, 9):
        damage = f"MVS_P2_DS{k}_NM_Z_01"
        day = "02" if k % 2 == 1 else "01"
        test_records = [damage, NORMAL[day]]
        test = frame["source_record"].isin(test_records)
        plan.append({
            "model_folder": f"heldout_DS{k}+UDS_{day}",
            "test_label": f"DS{k} + UDS_{day}({NORMAL_DATE[day]})",
            "train_label": f"나머지 손상 7개 + UDS_{'01' if day == '02' else '02'}",
            "n_train_windows": int((~test).sum()), "n_test_windows": int(test.sum()),
            "train_records": [r for r in records if r not in test_records],
            "test_records": test_records,
        })
    return plan


def inner_groups(frame: pd.DataFrame) -> pd.Series:
    """Inner split groups: one per damage recording; each normal recording cut into 4 time blocks."""
    group = frame["source_record"].astype(str).copy()
    for rec in frame.loc[frame["damage_stage"] == "UDS", "source_record"].unique():
        idx = frame.index[frame["source_record"] == rec]
        order = frame.loc[idx, "window_start_us"].rank(method="first").to_numpy() - 1
        group[idx] = [f"{rec}#b{int(q)}" for q in np.floor(order * 4 / len(idx))]
    return group


DOC = __doc__.split("Design")[1].split("Usage")[0].strip()
SPEC = xe.Spec(
    name="hbta_nm", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN, design_doc=DOC,
    eval_label="손상 측정 1개 + 정상 측정 1개 빼기 (8번, 측정을 자르지 않음)",
    load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR,
)
SPEC_P95 = xe.Spec(
    name="hbta_nm", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN + "; threshold normal_p95 inner5",
    design_doc=DOC + "\nvariant: p95", eval_label="측정 빼기 8번 + 기준을 학습 정상 데이터 상위 5%로",
    load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR, threshold_mode="normal_p95", inner_groups=inner_groups,
)
SPEC_AVG3_P95 = xe.Spec(
    name="hbta_nm_avg3", input_files=INPUT_FILES, features=FEATURES,
    design=EVAL_DESIGN + "; features mean of 3 consecutive windows; threshold normal_p95 inner5",
    design_doc=DOC + "\nvariant: avg3_p95", eval_label="Window 3개 평균 + 측정 빼기 8번 + 기준을 학습 정상 데이터 상위 5%로",
    load=partial(load_frames, k=3), plan=fold_plan, sensor_dir=DATA_DIR,
    threshold_mode="normal_p95", inner_groups=inner_groups,
)
VARIANTS = {"base": SPEC, "p95": SPEC_P95, "avg3_p95": SPEC_AVG3_P95}


def main(argv=None) -> int:
    return xe.main(VARIANTS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
