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

Usage: python -m src.public_grade.xgb_nm_experiment
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.public_grade import loader
from src.public_grade import xgb_engine as xe

DATA_DIR = Path("data/raw/ai1_public_1.1")
FEATURES = list(xe.BASE_FEATURES)
INPUT_FILES = [DATA_DIR / "hbta_anomaly_results_raw.csv", DATA_DIR / "hbta_anomaly_results_deg.csv"]
EVAL_DESIGN = "leave-one-damage-recording-out with one normal recording (alternating day) v1, held = degraded hold_main, p > 0.5"
NORMAL = {"01": "MVS_P2_UDS_NM_Z_01", "02": "MVS_P2_UDS_NM_Z_02"}
NORMAL_DATE = {"01": "9/27", "02": "9/22"}


def load_frames() -> dict:
    raw, deg = loader.load_pair(DATA_DIR, "hbta")
    held = deg["held_main"].to_numpy()
    out = {}
    for kind, frame in (("raw", raw), ("deg", deg)):
        frame = frame.copy()
        frame["held"] = held
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


SPEC = xe.Spec(
    name="hbta_nm", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN,
    design_doc=__doc__.split("Design")[1].split("Usage")[0].strip(),
    eval_label="손상 측정 1개 + 정상 측정 1개 빼기 (8번, 측정을 자르지 않음)",
    load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR,
)


def main(argv=None) -> int:
    return xe.main(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
