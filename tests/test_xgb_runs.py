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
