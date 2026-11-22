"""Synthetic tests for the Truss v4 evaluation (no public data needed).

The numbers here are made up to check the arithmetic and the delivery checks.
They are never used as performance evidence.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.public_grade import grade_v3 as gv
from src.public_grade import truss_v4 as t4

UTC = timezone.utc


def _epoch_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1e6)


def _values(rng, design, n, kind):
    """normal: random around 0; flat: the normal centre (always within p95); danger: far away."""
    ar_cols, tac_cols = list(design.ar_columns), list(design.tac_columns)
    if kind == "normal":
        ar = rng.normal(size=(n, len(ar_cols)))
        tac = rng.uniform(0.1, 0.3, size=(n, len(tac_cols)))
    elif kind == "flat":
        ar = np.zeros((n, len(ar_cols)))
        tac = np.full((n, len(tac_cols)), 0.2)
    else:
        ar = np.full((n, len(ar_cols)), 10.0)
        tac = np.full((n, len(tac_cols)), 0.9)
    return pd.DataFrame(np.hstack([ar, tac]), columns=ar_cols + tac_cols)


KIND_OF_ROLE = {"기준 적합": "normal", "정상 기준": "normal", "같은 계절 오경보": "flat", "오경보 확인": "flat",
                "작은 손상": "flat", "위험 탐지": "danger", "파단 중": "danger"}


def lumo_frame(seed=0, blocks=2, shift_days=None, drop=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    design = t4.DESIGNS["lumo"]
    parts = []
    for span in t4.LUMO_SPANS:
        if span.name == drop:
            continue
        start = t4._utc(span.start) + timedelta(days=(shift_days or {}).get(span.name, 0))
        for b in range(blocks):
            block_start = start + timedelta(seconds=600 * b)
            n = t4.LUMO_WINDOWS_PER_BLOCK
            p = _values(rng, design, n, KIND_OF_ROLE[span.role])
            p["window_start_us"] = [_epoch_us(block_start + timedelta(seconds=30 * i)) for i in range(n)]
            p["source_record"] = f"SHMTS_{(block_start + timedelta(seconds=600)):%Y%m%d_%H%M%S}"
            p["damage_stage"], p["ai1_split"], p["cv_group"] = span.stage, span.split, span.name
            parts.append(p)
    return _finish_frame(pd.concat(parts, ignore_index=True), "lumo")


def van_names():
    t0 = datetime(2023, 2, 20, tzinfo=UTC)
    pre = [t0 + timedelta(hours=7 * i) for i in range(t4.VAN_N_PRE)]
    dur = datetime(2023, 3, 9, 23, 45, 25, tzinfo=UTC)
    post = [datetime(2023, 3, 10, tzinfo=UTC) + timedelta(hours=5 * i) for i in range(t4.VAN_N_POST)]
    return [(d, f"{d:%Y_%m_%dT%H_%M_%S}") for d in pre + [dur] + post]


def van_frame(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    design = t4.DESIGNS["vanersborg"]
    names = van_names()
    plan = {e: (s, st, r) for e, s, st, r in t4.vanersborg_events(pd.DataFrame({"source_record": [n for _, n in names]}))}
    parts = []
    for start, name in names:
        split, stage, role = plan[name]
        n = 8
        p = _values(rng, design, n, KIND_OF_ROLE[role])
        p["window_start_us"] = [_epoch_us(start) + int(i * 10.24e6) for i in range(n)]
        p["source_record"], p["damage_stage"], p["ai1_split"] = name + ".csv", stage, split
        p["cv_group"] = role
        parts.append(p)
    return _finish_frame(pd.concat(parts, ignore_index=True), "vanersborg")


def _finish_frame(frame, dataset):
    frame["experiment_id"] = [f"{dataset}_{i:05d}" for i in range(len(frame))]
    frame["label_status"] = "확인용"
    frame["hold_main"] = False
    return frame


def write_delivery(tmp_path, frame, dataset, raw_frame=None):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    frame.to_csv(t4.result_path(data, dataset, "deg"), index=False, encoding="utf-8-sig")
    (raw_frame if raw_frame is not None else frame.drop(columns=["hold_main"])).to_csv(
        t4.result_path(data, dataset, "raw"), index=False, encoding="utf-8-sig")
    return data


def run(tmp_path, data, dataset):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("synthetic prereg", encoding="utf-8")
    return t4.run(data, dataset, out_dir=tmp_path / "out", summary_dir=tmp_path / "results", prereg=prereg)


# ------------------------------------------------------------------ design and delivery checks

def test_lumo_design_has_31_spans_with_preregistered_roles():
    spans = t4.LUMO_SPANS
    assert len(spans) == 31 and len({s.name for s in spans}) == 31
    count = pd.Series([s.role for s in spans]).value_counts().to_dict()
    assert count == {"기준 적합": 2, "정상 기준": 4, "같은 계절 오경보": 1, "오경보 확인": 12,
                     "위험 탐지": 6, "작은 손상": 6}
    assert {s.split for s in spans if s.role == "기준 적합"} == {"train"}
    assert {s.split for s in spans if s.role == "정상 기준"} == {"ai2_calib"}


def test_span_shift_inside_fallback_and_outside():
    span = t4.Span("X_A", "UDS", "test", "오경보 확인", "2021-01-01 00:00", +1)
    t0 = t4._utc(span.start)
    inside = [_epoch_us(t0 - timedelta(seconds=540)), _epoch_us(t0 + timedelta(hours=8, minutes=59))]
    assert t4.span_shift(span, inside) == 0
    assert t4.span_shift(span, [_epoch_us(t0 + timedelta(days=2, hours=1))]) == 2
    assert t4.span_shift(span, [_epoch_us(t0 - timedelta(days=1))]) is None      # A spans only move later
    assert t4.span_shift(span, [_epoch_us(t0 + timedelta(hours=9))]) is None    # end is exclusive


def test_check_lumo_accepts_design_and_records_shift():
    shifts = t4.check_lumo(lumo_frame(shift_days={"N3_B": -1}))
    assert shifts["N3_B"] == -1 and shifts["N2_A"] == 0


def test_check_lumo_stops_on_missing_span():
    with pytest.raises(t4.DeliveryError, match="DAM4_111_B"):
        t4.check_lumo(lumo_frame(drop="DAM4_111_B"))


def test_check_lumo_stops_on_wrong_split():
    frame = lumo_frame()
    frame.loc[frame["cv_group"] == "N2_A", "ai1_split"] = "ai2_calib"
    with pytest.raises(t4.DeliveryError, match="N2_A"):
        t4.check_lumo(frame)


def test_vanersborg_event_roles():
    plan = t4.vanersborg_events(pd.DataFrame({"source_record": [n for _, n in van_names()]}))
    roles = pd.Series([r for *_, r in plan]).value_counts().to_dict()
    assert roles == {"기준 적합": 20, "정상 기준": 20, "오경보 확인": 17, "파단 중": 1, "위험 탐지": 6}
    assert plan[57][0] == t4.VAN_DUR and plan[57][2] == "DUR"
    assert all(p[2] == "POST" for p in plan[58:])


def test_vanersborg_stops_when_damage_event_is_not_58th():
    names = [n for _, n in van_names()]
    names[0] = "2023_03_10T23_00_00"          # one pre event moved after the damage event
    with pytest.raises(t4.DeliveryError, match="expected"):
        t4.vanersborg_events(pd.DataFrame({"source_record": names}))


# ------------------------------------------------------------------ rule and metrics

def test_tac_mean_is_missing_when_one_axis_is_missing():
    frame = lumo_frame()
    frame["held"] = False
    frame.loc[0, "tac_a_b_x"] = np.nan
    ind, _ = t4.indicators(frame, t4.DESIGNS["lumo"])
    assert np.isnan(ind.loc[0, "tac"])
    assert ind.loc[1, "tac"] == pytest.approx(frame.loc[1, ["tac_a_b_x", "tac_a_b_y"]].mean())


def test_held_fit_windows_do_not_change_ar_reference():
    frame = lumo_frame()
    frame["held"] = False
    i = frame.index[frame["ai1_split"] == "train"][0]
    without, _ = t4.indicators(frame.drop(index=i), t4.DESIGNS["lumo"])   # row i absent
    frame.loc[i, list(t4.DESIGNS["lumo"].ar_columns)] = 1e6
    frame.loc[i, "held"] = True                                            # row i present but held
    after, _ = t4.indicators(frame, t4.DESIGNS["lumo"])
    keep = frame.index != i
    assert np.allclose(without["ar"], after.loc[keep, "ar"])


def _w(grades, judged):
    return pd.DataFrame({"grade": grades, "judged": judged, "abnormal": pd.Series(grades).isin(t4.ABNORMAL)})


def test_rate_counts_only_judged_windows():
    w = _w(["경고", "정상", gv.HELD, "주의", "위험"], [True, True, False, False, True])
    r = t4.rate(w, [True] * 5)
    assert r == {"windows": 5, "held": 1, "undecided": 1, "judged": 3, "abnormal": 2, "rate": pytest.approx(2 / 3)}


def test_decide_thresholds_and_tentative():
    ok = t4.decide({"rate": 0.05, "judged": 200}, {"rate": 0.95, "judged": 200})
    assert ok == {"false_alarm_pass": True, "detection_pass": True, "pass": True, "tentative": False}
    bad = t4.decide({"rate": 0.06, "judged": 100}, {"rate": 0.95, "judged": 300})
    assert bad["pass"] is False and bad["false_alarm_pass"] is False and bad["tentative"] is True
    assert t4.decide({"rate": None, "judged": 0}, {"rate": 1.0, "judged": 300})["pass"] is False


def test_auc_perfect_and_ties():
    assert t4.auc([0, 1, 2], [3, 4]) == 1.0
    assert t4.auc([1, 1], [1]) == 0.5
    assert t4.auc([], [1]) is None


# ------------------------------------------------------------------ end to end (synthetic)

def test_lumo_end_to_end_separated_synthetic(tmp_path):
    data = write_delivery(tmp_path, lumo_frame(blocks=3), "lumo")
    s = run(tmp_path, data, "lumo")
    r = s["result"]
    n_fa = 12 * 3 * (18 - 2)                    # 12 spans x 3 blocks x windows with persistence
    assert r["false_alarm"]["judged"] == n_fa and r["false_alarm"]["rate"] == 0.0
    assert r["detection"]["judged"] == 6 * 3 * 16 and r["detection"]["rate"] == 1.0   # 288 >= 200
    assert r["false_alarm"]["undecided"] == 12 * 3 * 2   # first two windows of each block
    assert r["pass"] is True and r["tentative"] is False
    assert r["by_stage"]["DAM6_111"]["grades"]["위험"] == 1.0
    assert r["auc"]["deg"]["ar"] == 1.0
    assert s["inputs"] and s["prereg_sha256"]
    w = pd.read_csv(tmp_path / "out" / "lumo_deg_grades.csv", encoding="utf-8-sig")
    assert set(w["status"]) == {t4.STATUS}
    assert w["reason"].str.contains("ML1").any()


def test_lumo_held_channel_makes_window_held(tmp_path):
    frame = lumo_frame()
    i = frame.index[frame["cv_group"] == "DAM6_111_A"][5]
    frame.loc[i, "hold_main"] = True
    held_id = frame.loc[i, "experiment_id"]
    data = write_delivery(tmp_path, frame, "lumo")
    s = run(tmp_path, data, "lumo")
    assert s["result"]["detection"]["held"] == 1
    w = pd.read_csv(tmp_path / "out" / "lumo_deg_grades.csv", encoding="utf-8-sig").set_index("experiment_id")
    assert w.loc[held_id, "grade"] == gv.HELD
    assert (w["grade"] == gv.HELD).sum() == 1


def test_run_refuses_second_run(tmp_path):
    data = write_delivery(tmp_path, lumo_frame(), "lumo")
    run(tmp_path, data, "lumo")
    with pytest.raises(SystemExit, match="run once"):
        run(tmp_path, data, "lumo")


def test_vanersborg_end_to_end_separated_synthetic(tmp_path):
    data = write_delivery(tmp_path, van_frame(), "vanersborg")
    s = run(tmp_path, data, "vanersborg")
    r = s["result"]
    assert r["false_alarm"]["judged"] == 17 * 6 and r["false_alarm"]["rate"] == 0.0
    assert r["detection"]["judged"] == 6 * 6 and r["detection"]["rate"] == 1.0
    assert r["pass"] is True and r["tentative"] is True        # 36 danger windows < 200
    assert r["events_any_danger"]["위험 탐지"] == {"events": 6, "with_danger": 6}
    assert r["events_any_danger"]["오경보 확인"]["with_danger"] == 0
    assert len(r["during_event_grades"]) == 8


def test_load_stops_on_missing_column(tmp_path):
    frame = lumo_frame().drop(columns=["tac_a_b_y"])
    data = write_delivery(tmp_path, frame, "lumo")
    with pytest.raises(t4.DeliveryError, match="tac_a_b_y"):
        t4.load(data, t4.DESIGNS["lumo"])


def test_check_lumo_stops_on_19_windows_per_block():
    frame = lumo_frame()
    first = frame[frame["cv_group"] == "N2_A"].iloc[[-1]].copy()
    first["window_start_us"] += 30_000_000
    first["experiment_id"] = "extra"
    with pytest.raises(t4.DeliveryError, match="more than 18"):
        t4.check_lumo(pd.concat([frame, first], ignore_index=True))
