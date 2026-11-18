"""Load and validate AI1-public raw/degraded result pairs (prereg 1, 2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("ponte_moesa", "hbta", "old_ada")
KINDS = ("raw", "deg")
KEY = ["experiment_id", "window_start_us"]
LABEL_COLUMNS = [
    "damage_stage", "label_status", "ai1_split", "cv_group", "subset",
]
OPTIONAL_LABEL_COLUMNS = ("cv_group", "subset")
FORBIDDEN_COLUMNS =("confidence", "anomaly_probability")
AI1_RESULT_CATEGORIES = ("AI① 결과", "AI① v2 결과")
RAW_HOLD_RMS = 0.00063

HOLD_COLUMNS = {
    ("ponte_moesa", "deg"): {"main": "hold_flag_4_2", "session": "hold_flag_2_3"},
    ("ponte_moesa", "raw"): {"main": "raw_hold_flag_4_2", "session": "raw_hold_flag_2_3"},
    ("hbta", "deg"): {"main": "hold_flag_a_b"},
    ("old_ada", "deg"): {"main": "hold_flag_a_b"},
}
RAW_RMS_HOLD = {
    ("hbta", "raw"): ("raw_rms_acc01_z", "raw_rms_acc02_z"),
    ("old_ada", "raw"): ("raw_rms_acc01_z", "raw_rms_acc02_z"),
}


class DeliveryError(ValueError):
    """Raised when a delivery violates the agreed file contract."""


def result_path(data_dir: Path, dataset: str, kind: str) -> Path:
    return Path(data_dir) / f"{dataset}_anomaly_results_{kind}.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise DeliveryError(f"missing delivery file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def held_from_flag(series: pd.Series) -> pd.Series:
    """True unless the flag is explicitly False (blank = not evaluable)."""
    text = series.astype("string").str.strip().str.lower()
    return (text != "false").fillna(True).astype(bool)


def ai1_result_columns(guide: pd.DataFrame) -> list[str]:
    category = guide["AI② 사용 구분"].astype(str)
    mask = category.apply(
        lambda value: any(value.startswith(c) for c in AI1_RESULT_CATEGORIES)
    )
    return guide.loc[mask, "열 이름"].astype(str).str.strip().tolist()


def add_hold_columns(frame: pd.DataFrame, dataset: str, kind: str) -> pd.DataFrame:
    frame = frame.copy()
    flags = HOLD_COLUMNS.get((dataset, kind), {})
    for name, column in flags.items():
        if column not in frame:
            raise DeliveryError(f"{dataset}_{kind}: missing hold column {column}")
        frame[f"held_{name}"] = held_from_flag(frame[column])
    if (dataset, kind) in RAW_RMS_HOLD:
        held = pd.Series(False, index=frame.index)
        for column in RAW_RMS_HOLD[(dataset, kind)]:
            values = pd.to_numeric(frame[column], errors="coerce")
            held |= values.isna() | (values < RAW_HOLD_RMS)
        frame["held_main"] = held
    if "held_main" not in frame:
        raise DeliveryError(f"{dataset}_{kind}: no main hold rule")
    return frame


def clear_train_ai1_results(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Blank AI1 score/decision columns on AI1 training rows (contract 8)."""
    frame = frame.copy()
    train = frame["ai1_split"] == "train"
    for column in columns:
        if column in frame:
            frame[column] = frame[column].astype(object)
            frame.loc[train, column] = np.nan
    return frame


def validate_pair(raw: pd.DataFrame, deg: pd.DataFrame, dataset: str) -> None:
    if len(raw) != len(deg):
        raise DeliveryError(f"{dataset}: raw/deg row count differs")
    for column in KEY + LABEL_COLUMNS:
        if column not in raw or column not in deg:
            if column in OPTIONAL_LABEL_COLUMNS and column not in raw and column not in deg:
                continue
            raise DeliveryError(f"{dataset}: missing column {column}")
        left = raw[column].astype("string").fillna("")
        right = deg[column].astype("string").fillna("")
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise DeliveryError(f"{dataset}: raw/deg differ in {column}")
    if raw.duplicated(KEY).any():
        raise DeliveryError(f"{dataset}: duplicate window keys")


def load_pair(data_dir: Path, dataset: str, guide: pd.DataFrame | None = None):
    """Return (raw, deg) frames with hold columns and train AI1 results blanked."""
    if dataset not in DATASETS:
        raise DeliveryError(f"unknown dataset {dataset}")
    if guide is None:
        guide = read_csv(Path(data_dir) / "column_guide.csv")
    blocked = ai1_result_columns(guide)
    frames = {}
    for kind in KINDS:
        frame = read_csv(result_path(data_dir, dataset, kind))
        frame = frame.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in frame])
        frame = clear_train_ai1_results(frame, blocked)
        frames[kind] = add_hold_columns(frame, dataset, kind)
    validate_pair(frames["raw"], frames["deg"], dataset)
    for kind in KINDS:
        for column in OPTIONAL_LABEL_COLUMNS:
            if column not in frames[kind]:
                frames[kind][column] = np.nan
    return frames["raw"], frames["deg"]


def file_digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
