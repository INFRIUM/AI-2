"""Run bookkeeping of the XGBoost development experiment (no model training here)."""

import pandas as pd

from src.public_grade import xgb_sm_experiment as xs


def test_new_fingerprint_creates_named_run(tmp_path):
    run_id, is_new = xs.resolve_run(tmp_path / "index.csv", "abcdef1234567890", "hbta_sm", "20261119")
    assert is_new and run_id == "hbta_sm_20261119_abcdef12"


def test_same_fingerprint_reuses_existing_run(tmp_path):
    index = tmp_path / "index.csv"
    xs.update_index(index, {"학습 건": "hbta_sm_20261119_abcdef12", "입력 지문": "abcdef1234567890"})
    run_id, is_new = xs.resolve_run(index, "abcdef1234567890", "hbta_sm", "20261201")
    assert not is_new and run_id == "hbta_sm_20261119_abcdef12"     # later date does not create a duplicate


def test_index_accumulates_and_replaces_same_run(tmp_path):
    index = tmp_path / "index.csv"
    xs.update_index(index, {"학습 건": "a_20261119_11111111", "입력 지문": "1", "원본 AUC": 0.5})
    xs.update_index(index, {"학습 건": "z24_20261201_22222222", "입력 지문": "2", "원본 AUC": 0.9})
    xs.update_index(index, {"학습 건": "a_20261119_11111111", "입력 지문": "1", "원본 AUC": 0.6})
    table = pd.read_csv(index, encoding="utf-8-sig", dtype=str)
    assert table["학습 건"].tolist() == ["a_20261119_11111111", "z24_20261201_22222222"]
    assert table.loc[0, "원본 AUC"] == "0.6"


def test_label_frame_custom_normal_grade_and_exclusion():
    from src.public_grade import xgb_engine as xe
    f = pd.DataFrame({"damage_stage": ["INT", "DMG1", "DMG3", "RCV"], "held": [False, False, True, False]})
    out = xe.label_frame(f, normal_stage="INT", grade_of={"INT": "정상", "DMG1": "주의", "DMG3": "위험"},
                         exclude_stages=("RCV",))
    assert out["damage_stage"].tolist() == ["INT", "DMG1"]          # held DMG3 and excluded RCV dropped
    assert out["damaged"].tolist() == [0, 1] and out["grade"].tolist() == ["정상", "주의"]


def test_oldada_folds_cover_each_pass_once_and_stratify():
    from src.public_grade import xgb_oldada_experiment as xo
    stages = ["INT"] * 10 + ["DMG1"] * 12 + ["DMG3"] * 10
    cond = ["VBA_40"] * 5 + ["VBA_30"] * 5 + ["VBA_40"] * 22
    f = pd.DataFrame({"damage_stage": stages, "condition": cond, "source_record": [f"r{i:02d}" for i in range(32)]})
    fold = xo.fold_of(f)
    assert set(fold) == set(range(xo.N_FOLDS))
    for stage in ("DMG1", "DMG3"):
        counts = fold[f["damage_stage"] == stage].value_counts()
        assert counts.max() - counts.min() <= 1                      # spread evenly over folds
    plan = xo.fold_plan(f)
    tested = sorted(r for p in plan for r in p["test_records"])
    assert tested == sorted(f["source_record"])                      # every pass evaluated exactly once
