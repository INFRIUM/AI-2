"""Unit tests for src.public_grade.

All numbers in this file are hand-made fixtures that exercise arithmetic and
contract rules only. They are not measurements, training data, baselines, or
performance evidence.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_grade import evaluate as ev
from src.public_grade import grading as g
from src.public_grade import loader


class HoldFlagTests(unittest.TestCase):
    def test_blank_and_true_are_held(self):
        flags = pd.Series(["True", "False", None, True, False, ""])
        self.assertEqual(loader.held_from_flag(flags).tolist(), [True, False, True, True, False, True])

    def test_raw_rms_rule_for_truss(self):
        frame = pd.DataFrame({
            "raw_rms_acc01_z": [0.001, 0.0005, 0.002, None],
            "raw_rms_acc02_z": [0.001, 0.002, 0.0006, 0.002],
        })
        out = loader.add_hold_columns(frame, "hbta", "raw")
        self.assertEqual(out["held_main"].tolist(), [False, True, True, True])


class TrainScoreBlockTests(unittest.TestCase):
    def test_ai1_result_columns_follow_guide(self):
        guide = pd.DataFrame({
            "열 이름": ["anomaly_score", "v2_prediction", "tac_4_2_z", "acc01_rms_z"],
            "AI② 사용 구분": ["AI① 결과", "AI① v2 결과 (참고)", "AI② 등급 지표", "Feature"],
        })
        self.assertEqual(loader.ai1_result_columns(guide), ["anomaly_score", "v2_prediction"])

    def test_train_rows_are_blanked_only(self):
        frame = pd.DataFrame({
            "ai1_split": ["train", "val", "test"],
            "anomaly_score": [0.9, 0.2, 0.3],
        })
        out = loader.clear_train_ai1_results(frame, ["anomaly_score"])
        self.assertTrue(pd.isna(out.loc[0, "anomaly_score"]))
        self.assertEqual(out.loc[1, "anomaly_score"], 0.2)


def _write_pair(directory: Path, raw: pd.DataFrame, deg: pd.DataFrame) -> None:
    raw.to_csv(directory / "old_ada_anomaly_results_raw.csv", index=False, encoding="utf-8-sig")
    deg.to_csv(directory / "old_ada_anomaly_results_deg.csv", index=False, encoding="utf-8-sig")


class LoadPairTests(unittest.TestCase):
    def _frames(self):
        base = pd.DataFrame({
            "experiment_id": ["E1", "E2"],
            "window_start_us": [1, 2],
            "damage_stage": ["INT", "INT"],
            "label_status": ["확실", "확실"],
            "ai1_split": ["train", "test"],
            "max_abs_z": [1.0, 2.0],
            "confidence": [0.5, 0.5],
            "raw_rms_acc01_z": [0.1, 0.1],
            "raw_rms_acc02_z": [0.1, 0.1],
            "hold_flag_a_b": [False, False],
        })
        return base.copy(), base.copy()

    def test_load_blocks_train_scores_and_drops_forbidden(self):
        guide = pd.DataFrame({"열 이름": ["max_abs_z"], "AI② 사용 구분": ["AI① 결과"]})
        raw, deg = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            _write_pair(Path(tmp), raw, deg)
            r, d = loader.load_pair(Path(tmp), "old_ada", guide)
        self.assertTrue(pd.isna(d.loc[0, "max_abs_z"]))
        self.assertEqual(d.loc[1, "max_abs_z"], 2.0)
        self.assertNotIn("confidence", d)
        self.assertIn("cv_group", d)

    def test_key_mismatch_stops(self):
        guide = pd.DataFrame({"열 이름": [], "AI② 사용 구분": []})
        raw, deg = self._frames()
        deg.loc[1, "window_start_us"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            _write_pair(Path(tmp), raw, deg)
            with self.assertRaises(loader.DeliveryError):
                loader.load_pair(Path(tmp), "old_ada", guide)


class GradingTests(unittest.TestCase):
    T = (0.0, 1.0, 2.0, 3.0, 4.0)

    def test_grade_boundaries(self):
        self.assertEqual(g.grade(1.0, self.T), "정상")
        self.assertEqual(g.grade(1.0001, self.T), "주의")
        self.assertEqual(g.grade(2.0, self.T), "주의")
        self.assertEqual(g.grade(3.5, self.T), "위험")

    def test_score_is_piecewise_linear_and_clipped(self):
        self.assertEqual(g.score(0.5, self.T), 12.5)
        self.assertEqual(g.score(2.5, self.T), 62.5)
        self.assertEqual(g.score(9.0, self.T), 100.0)
        self.assertEqual(g.score(-1.0, self.T), 0.0)

    def test_non_monotonic_thresholds_give_no_grade(self):
        bad = (0.0, 1.0, 0.5, 3.0, 4.0)
        self.assertFalse(g.thresholds_valid(bad))
        self.assertIsNone(g.grade(2.0, bad))
        self.assertIsNone(g.score(2.0, bad))

    def test_upper_thresholds_use_pooled_medians(self):
        t2, t3, t4, med = g.upper_thresholds({"주의": [1, 3], "경고": [5], "위험": [9, 11]})
        self.assertEqual((t2, t3, t4), (3.5, 7.5, 10.0))

    def test_lower_thresholds(self):
        t0, t1 = g.lower_thresholds(list(range(1, 21)))
        self.assertEqual(t0, 10.5)
        self.assertAlmostEqual(t1, np.percentile(range(1, 21), 95))

    def test_mahalanobis_matches_manual(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
        target = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 0.0]})
        d, _ = g.mahalanobis(ref, target, ["a", "b"])
        x = np.array([1.0, 2.0]) - ref.mean().to_numpy()
        manual = np.sqrt(x @ np.linalg.inv(np.cov(ref.to_numpy(), rowvar=False)) @ x)
        self.assertAlmostEqual(d.iloc[0], manual)
        self.assertTrue(np.isnan(d.iloc[1]))

    def test_standardized_euclid(self):
        ref = pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 4.0]})
        d, _ = g.standardized_euclid(ref, pd.DataFrame({"a": [1.0], "b": [2.0 + 2 * np.sqrt(2)]}), ["a", "b"])
        self.assertAlmostEqual(d.iloc[0], 1.0)

    def test_bootstrap_is_seeded(self):
        values = list(range(30))
        self.assertEqual(g.t1_uncertainty(values), g.t1_uncertainty(values))


def _box_fixture(normal_values, a_values, b_values) -> pd.DataFrame:
    """Build a prepared Box-like frame. a/b_values: dict stage -> list."""
    rows = []
    for i, v in enumerate(np.linspace(0.0, 1.0, 40)):
        rows.append(("C", i, "ai2_calib", "DS2", "확실", None, v))
    for i, v in enumerate(normal_values):
        rows.append(("N", i, "test", "DS2", "확실", None, v))
    for group, values in (("A", a_values), ("B", b_values)):
        for stage, vals in values.items():
            for i, v in enumerate(vals):
                rows.append((f"{stage}{group}", i, "test", stage, "확실", group, v))
    frame = pd.DataFrame(rows, columns=[
        "experiment_id", "window_start_us", "ai1_split", "damage_stage",
        "label_status", "cv_group", "x"])
    frame["held_main"] = False
    return ev.prepare(frame, frame["x"], "held_main")


class EvaluateBridgeTests(unittest.TestCase):
    def test_well_separated_fixture_passes(self):
        frame = _box_fixture(
            normal_values=[0.5] * 30,
            a_values={"DS3": [2.0] * 5, "DS5": [4.0] * 5, "DS7": [8.0] * 5},
            b_values={"DS3": [2.2] * 5, "DS5": [4.2] * 5, "DS7": [8.2] * 5},
        )
        summary, windows = ev.evaluate_bridge(frame, ev.BOX)
        cond = summary["conditions"]
        self.assertTrue(all(c["pass"] for c in cond.values()), cond)
        self.assertEqual(ev.verdict(cond), "통과 (공개 데이터 기준. API 연결 안 함)")
        damage = windows[windows["damage_stage"] != "DS2"]
        self.assertTrue(damage["grade_basis"].str.startswith("held-out").all())

    def test_folds_use_other_group_only(self):
        frame = _box_fixture(
            normal_values=[0.5] * 30,
            a_values={"DS3": [2.0] * 3, "DS5": [4.0] * 3, "DS7": [8.0] * 3},
            b_values={"DS3": [20.0] * 3, "DS5": [40.0] * 3, "DS7": [80.0] * 3},
        )
        summary, _ = ev.evaluate_bridge(frame, ev.BOX)
        folds = {f["fold"]: f["thresholds"] for f in summary["folds"]}
        # evaluating A uses thresholds fitted on B, and vice versa
        self.assertEqual(folds["cv_group=A"][2:], [30.0, 60.0, 80.0])
        self.assertEqual(folds["cv_group=B"][2:], [3.0, 6.0, 8.0])
        self.assertFalse(summary["conditions"]["5_generalization"]["pass"])

    def test_false_alarm_and_order_failures(self):
        frame = _box_fixture(
            normal_values=[0.5] * 27 + [5.0] * 3,
            a_values={"DS3": [4.0] * 5, "DS5": [2.0] * 5, "DS7": [8.0] * 5},
            b_values={"DS3": [4.0] * 5, "DS5": [2.0] * 5, "DS7": [8.0] * 5},
        )
        summary, _ = ev.evaluate_bridge(frame, ev.BOX)
        cond = summary["conditions"]
        self.assertEqual(cond["1_false_alarm"]["exceed"], 3)
        self.assertFalse(cond["1_false_alarm"]["pass"])
        self.assertFalse(cond["3_grade_order"]["pass"])
        self.assertTrue(ev.verdict(cond).startswith("출력 불가"))

    def test_held_windows_are_excluded(self):
        frame = _box_fixture(
            normal_values=[0.5] * 30 + [9.0] * 5,
            a_values={"DS3": [2.0] * 5, "DS5": [4.0] * 5, "DS7": [8.0] * 5},
            b_values={"DS3": [2.2] * 5, "DS5": [4.2] * 5, "DS7": [8.2] * 5},
        )
        frame.loc[frame["x"] == 9.0, "held"] = True
        frame["valid"] = ~frame["held"] & frame["value"].notna()
        summary, windows = ev.evaluate_bridge(frame, ev.BOX)
        self.assertEqual(summary["conditions"]["1_false_alarm"]["exceed"], 0)
        self.assertEqual(summary["counts"]["DS2 test"]["held"], 5)
        self.assertEqual((windows["grade"] == "판정 보류").sum(), 5)

    def test_hbta_cause_rules(self):
        deg = {"1_false_alarm": {"pass": False}, "3_grade_order": {"pass": False},
               "5_generalization": {"pass": False}, "2_danger_detection": {"pass": True}}
        raw_z = {"1_false_alarm": {"pass": True}, "3_grade_order": {"pass": False},
                 "5_generalization": {"pass": False}}
        raw_yz = {"3_grade_order": {"pass": True}, "5_generalization": {"pass": False}}
        causes = ev.hbta_cause(deg, raw_z, raw_yz)
        self.assertTrue(causes["1_false_alarm"].startswith("센서 잡음"))
        self.assertTrue(causes["3_grade_order"].startswith("y축 손실"))
        self.assertTrue(causes["5_generalization"].startswith("지표 또는 데이터 한계"))
        self.assertNotIn("2_danger_detection", causes)


if __name__ == "__main__":
    unittest.main()
