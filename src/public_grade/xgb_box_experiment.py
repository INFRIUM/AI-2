"""XGBoost development experiment on Ponte Moesa (Box): raw vs degraded.

DEVELOPMENT ONLY. Ponte Moesa (AI1-public-1.1) was used in v1 and grade v3,
so every number is "결과 열람 후 실험", not performance evidence. Shared
training, outputs and run bookkeeping live in xgb_engine.py.

Design (fixed before the first run, 2026-11-20):
  data      Ponte Moesa, label_status = 확실 only (AMB_36 DS5 and AMB_37 DS6
            are 참고 and excluded). Windows usable in the degraded data
            (hold_main False): normal DS2 = AMB_16 (before work, 11/27) and
            AMB_18 (during work, 11/27 12:45-17:00); the 11/26-27 session
            recordings are all held (no sensor 4). Damage: DS3 AMB_27-30
            (11/28 a.m.), DS5 AMB_35 (11/28 p.m.), DS7 AMB_47 (11/29 a.m.)
  features  acc01 (sensor 4, 3/4L) and acc02 (sensor 2, L/2) x/y/z window
            statistics + cross_amp_ratio_L4_L2_z + cross_torsion_y_z_L2 (68).
            acc03 is excluded (no HW sensor at that position). No AR in the
            delivery for Ponte Moesa.
  grade     v1 map: DS2 정상, DS3 주의, DS5 경고, DS7 위험. Danger = DS7.
  split     6 folds. Damage recordings are never split: each fold holds out
            one damage recording + one normal unit. Normal units: AMB_18 cut
            into 5 time blocks (the first window of blocks 2-5 is dropped
            because 60 s windows overlap by 30 s) + AMB_16.
  tasks, params, metrics: same as the other experiments (xgb_engine)
  limits    every damage stage was measured on a different day (11/28, 11/29)
            and usable normal data is a single day (11/27, mostly with
            construction work), so a model cannot separate damage from the
            measurement day. Results show "how different from 11/27", not
            damage detection.

Variants (--variant): base (p > 0.5), p95 (threshold = 95th percentile of
out-of-fold probabilities of the training normal windows, inner split by unit).

Usage: python -m src.public_grade.xgb_box_experiment [--variant base|p95]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import loader
from src.public_grade import xgb_engine as xe

DATA_DIR = Path("data/raw/ai1_public_1.1")
STATS = xe.STATS
FEATURES = ([f"{s}_{k}_{a}" for s in ("acc01", "acc02") for a in ("x", "y", "z") for k in STATS]
            + ["cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2"])
INPUT_FILES = [DATA_DIR / "ponte_moesa_anomaly_results_raw.csv", DATA_DIR / "ponte_moesa_anomaly_results_deg.csv"]
EVAL_DESIGN = "ponte moesa leave-one-damage-recording-out with one normal unit (AMB_18 5 blocks + AMB_16) v1, 확실 only, held = degraded hold_main, p > 0.5"
GRADE_OF = {"DS2": "정상", "DS3": "주의", "DS5": "경고", "DS7": "위험"}
DAMAGE_RECORDS = ["AMB_27", "AMB_28", "AMB_29", "AMB_30", "AMB_35", "AMB_47"]
EXCLUDED_RECORDS = ("AMB_36", "AMB_37")    # 참고 recordings (AMB_36 has 3 windows marked 확실; whole recording excluded)
N_BLOCKS = 5


def block_normal(frame: pd.DataFrame) -> pd.DataFrame:
    """Cut AMB_18 into time blocks named AMB_18#b1..b5; drop the first window of
    blocks 2-5 (it overlaps the previous block by 30 s)."""
    frame = frame.copy()
    long = frame["source_record"] == "AMB_18"
    order = frame.loc[long, "window_start_us"].rank(method="first").to_numpy() - 1
    block = (np.floor(order * N_BLOCKS / long.sum()) + 1).astype(int)
    frame.loc[long, "source_record"] = [f"AMB_18#b{b}" for b in block]
    keep = pd.Series(True, index=frame.index)
    for b in range(2, N_BLOCKS + 1):
        idx = frame.index[frame["source_record"] == f"AMB_18#b{b}"]
        if len(idx):
            keep[frame.loc[idx, "window_start_us"].idxmin()] = False
    return frame[keep]


def load_frames() -> dict:
    raw, deg = loader.load_pair(DATA_DIR, "ponte_moesa")
    held = deg["held_main"].to_numpy()
    out = {}
    for kind, frame in (("raw", raw), ("deg", deg)):
        frame = frame.copy()
        frame["held"] = held
        frame = frame[(frame["label_status"] == "확실") & ~frame["source_record"].isin(EXCLUDED_RECORDS)]
        frame = block_normal(frame[~frame["held"]])
        out[kind] = xe.label_frame(frame, normal_stage="DS2", grade_of=GRADE_OF)
    keys = ["experiment_id", "window_start_us"]
    if not out["raw"][keys].reset_index(drop=True).equals(out["deg"][keys].reset_index(drop=True)):
        raise ValueError("raw and deg windows differ")
    return out


def fold_plan(frame: pd.DataFrame) -> list[dict]:
    normal_units = sorted(u for u in frame["source_record"].unique() if u.startswith("AMB_18#")) + ["AMB_16"]
    records = sorted(frame["source_record"].unique())
    plan = []
    for damage, normal in zip(DAMAGE_RECORDS, normal_units):
        test_records = [damage, normal]
        test = frame["source_record"].isin(test_records)
        stage = frame.loc[frame["source_record"] == damage, "damage_stage"].iloc[0]
        plan.append({
            "model_folder": f"heldout_{damage}+{normal.replace('#', '_')}",
            "test_label": f"{damage}({stage}) + {normal}",
            "train_label": "나머지 손상 측정 5개 + 나머지 정상 묶음 5개",
            "n_train_windows": int((~test).sum()), "n_test_windows": int(test.sum()),
            "train_records": [r for r in records if r not in test_records],
            "test_records": test_records,
        })
    return plan


DOC = __doc__.split("Design")[1].split("Usage")[0].strip()
COMMON = dict(input_files=INPUT_FILES, features=FEATURES, load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR,
              extra_columns=("subset",), normal_stage="DS2", danger_stages=("DS7",), grade_of=GRADE_OF,
              sensor_dataset="ponte_moesa")
SPEC = xe.Spec(name="ponte_moesa", design=EVAL_DESIGN, design_doc=DOC,
               eval_label="손상 측정 1개 + 정상 묶음 1개 빼기 (6번, 손상 측정은 자르지 않음)", **COMMON)
SPEC_P95 = xe.Spec(name="ponte_moesa", design=EVAL_DESIGN + "; threshold normal_p95 inner5", design_doc=DOC + "\nvariant: p95",
                   eval_label="측정 빼기 6번 + 기준을 학습 정상 데이터 상위 5%로", threshold_mode="normal_p95", **COMMON)
VARIANTS = {"base": SPEC, "p95": SPEC_P95}


def main(argv=None) -> int:
    return xe.main(VARIANTS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
