"""XGBoost development experiment on Old ADA (vehicle passes): raw vs degraded.

DEVELOPMENT ONLY. Old ADA (AI1-public-1.1) was used in v1 and grade v3, so
every number is "결과 열람 후 실험", not performance evidence. Shared
training, outputs and run bookkeeping live in xgb_engine.py.

Design (fixed before the first run, 2026-11-20):
  data      Old ADA: one free-decay window per vehicle pass (68 passes)
            normal INT 26 (40 km/h 10 + speed reference 30/50 km/h 16),
            DMG1 12 (T1 half cut), DMG2 10 (T1 cut), DMG3 10 (T2 cut + T1
            residual damage); RCV 10 (after repair, incomplete) excluded
  features  vertical (z) window features available in Old ADA: the z part of
            the shared feature set + damping ratio of both sensors (35).
            Vehicle speed is NOT an input: damage passes are all near 40 km/h,
            so speed would identify the label.
  grade     AI② definition for this experiment (Old ADA had no grade map),
            by the damage descriptions: INT 정상, DMG1 주의, DMG2 경고,
            DMG3 위험. Danger for detection = DMG3.
  split     a pass is never split. 5 folds stratified by stage (normal split
            into 40 km/h and 30/50 km/h), fixed seed; each fold holds out
            about 1/5 of the passes of every group.
  tasks, params, metrics: same as the other experiments (xgb_engine), plus
            false alarm by speed group (condition VBA_30/40/50)
  limits    passes of one stage were probably driven in one session, so a
            model can learn the session instead of damage; 58 passes only.

Variants (5-2, added 2026-11-20 before running them; --variant):
  base  the design above, decision p > 0.5 (run old_ada_20261120_f43d6505)
  p95   same models, decision threshold = 95th percentile of out-of-fold
        probabilities of the TRAINING normal passes (inner 5-fold split by
        pass inside the training data). Evaluated passes are never used.

Usage: python -m src.public_grade.xgb_oldada_experiment [--variant base|p95]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import loader
from src.public_grade import xgb_engine as xe

DATA_DIR = Path("data/raw/ai1_public_1.1")
FEATURES = [f for f in xe.BASE_FEATURES if f.endswith("_z")] + ["acc01_damping_z", "acc02_damping_z"]
INPUT_FILES = [DATA_DIR / "old_ada_anomaly_results_raw.csv", DATA_DIR / "old_ada_anomaly_results_deg.csv"]
EVAL_DESIGN = "old ada 5-fold stratified by pass group v1, RCV excluded, no speed input, p > 0.5"
GRADE_OF = {"INT": "정상", "DMG1": "주의", "DMG2": "경고", "DMG3": "위험"}
N_FOLDS = 5


def load_frames() -> dict:
    raw, deg = loader.load_pair(DATA_DIR, "old_ada")
    held = deg["held_main"].to_numpy()
    out = {}
    for kind, frame in (("raw", raw), ("deg", deg)):
        frame = frame.copy()
        frame["held"] = held
        out[kind] = xe.label_frame(frame, normal_stage="INT", grade_of=GRADE_OF, exclude_stages=("RCV",))
    keys = ["experiment_id", "window_start_us"]
    if not out["raw"][keys].equals(out["deg"][keys]):
        raise ValueError("raw and deg windows differ")
    return out


def pass_group(frame: pd.DataFrame) -> pd.Series:
    group = frame["damage_stage"].copy()
    normal = frame["damage_stage"] == "INT"
    group[normal] = np.where(frame.loc[normal, "condition"] == "VBA_40", "INT_40", "INT_speed")
    return group


def fold_of(frame: pd.DataFrame) -> pd.Series:
    """Assign every pass to a fold: shuffle each group with a fixed seed, deal round-robin."""
    rng = np.random.default_rng(xe.SEED)
    group = pass_group(frame)
    fold = pd.Series(-1, index=frame.index)
    for g in sorted(group.unique()):
        idx = frame.index[group == g].to_numpy()
        idx = idx[np.argsort(frame.loc[idx, "source_record"].to_numpy())]     # stable order before shuffling
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % N_FOLDS
    return fold


def fold_plan(frame: pd.DataFrame) -> list[dict]:
    fold = fold_of(frame)
    plan = []
    for k in range(N_FOLDS):
        test = fold == k
        counts = frame.loc[test, "damage_stage"].value_counts().reindex(["INT", "DMG1", "DMG2", "DMG3"]).fillna(0)
        plan.append({
            "model_folder": f"heldout_fold{k + 1}",
            "test_label": f"묶음 {k + 1} (" + ", ".join(f"{s} {int(n)}" for s, n in counts.items()) + ")",
            "train_label": "나머지 4개 묶음",
            "n_train_windows": int((~test).sum()), "n_test_windows": int(test.sum()),
            "train_records": sorted(frame.loc[~test, "source_record"]),
            "test_records": sorted(frame.loc[test, "source_record"]),
        })
    return plan


DOC = __doc__.split("Design")[1].split("Usage")[0].strip()
SPEC = xe.Spec(
    name="old_ada", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN,
    design_doc=DOC,
    eval_label="주행 단위 5묶음 나누기 (손상 상태·속도별 고르게, RCV 제외)",
    load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR, extra_columns=("speed_kmh",),
    normal_stage="INT", danger_stages=("DMG3",), grade_of=GRADE_OF, exclude_stages=("RCV",),
    sensor_dataset="old_ada",
)


SPEC_P95 = xe.Spec(
    name="old_ada", input_files=INPUT_FILES, features=FEATURES, design=EVAL_DESIGN + "; threshold normal_p95 inner5",
    design_doc=DOC + "\nvariant: p95", eval_label="주행 단위 5묶음 + 기준을 학습 정상 데이터 상위 5%로",
    load=load_frames, plan=fold_plan, sensor_dir=DATA_DIR, extra_columns=("speed_kmh",),
    normal_stage="INT", danger_stages=("DMG3",), grade_of=GRADE_OF, exclude_stages=("RCV",),
    sensor_dataset="old_ada", threshold_mode="normal_p95",
)
VARIANTS = {"base": SPEC, "p95": SPEC_P95}


def main(argv=None) -> int:
    return xe.main(VARIANTS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
