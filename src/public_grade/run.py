"""Terminal entry point: python -m src.public_grade.run --data-dir <dir>."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.public_grade import evaluate as ev
from src.public_grade import loader

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs" / "AI2_PUBLIC_GRADE_PREREG_v1.md"
DEFAULT_SUMMARY = ROOT / "results" / "public_grade" / "summary.json"
DEFAULT_WINDOWS = ROOT / "data" / "processed" / "public_grade" / "window_grades.csv"
CONDITION_NAMES = {
    "1_false_alarm": "1 정상 오경보율 ≤ 5%",
    "2_danger_detection": "2 위험 탐지율 ≥ 95%",
    "3_grade_order": "3 등급 순서",
    "4_extent_order": "4 손상 범위 순서",
    "5_generalization": "5 상태 일반화",
}


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def _drop_test(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["ai1_split"] != "test"].copy()


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _pct(rate):
    return "-" if rate is None else f"{rate * 100:.1f}%"


def print_bridge(title: str, summary: dict, provisional_note: str | None = None) -> None:
    print(f"\n[{title}]")
    if provisional_note:
        print(f"  ※ {provisional_note}")
    t = summary["thresholds_final"]
    print("  임계값(최종): " + "  ".join(f"{k}={_fmt(v)}" for k, v in t.items()))
    unc = summary["t1_uncertainty"]
    if unc:
        lo, hi = unc["p95_bootstrap_90ci"]
        print(f"  T1 불확실성: n={unc['n']} 최댓값={_fmt(unc['max'])} 상위3={[_fmt(v) for v in unc['top3']]} "
              f"부트스트랩 90%=[{_fmt(lo)}, {_fmt(hi)}]")
    print("  상태      전체  보류  유효  중앙값      잠정")
    for name, c in summary["counts"].items():
        stage = name.replace(" test", "")
        med = summary["stage_medians"].get(name, summary["stage_medians"].get(stage))
        print(f"  {name:<9}{c['total']:>5}{c['held']:>6}{c['valid']:>6}  {_fmt(med):<10}  "
              f"{'잠정' if c['provisional'] else ''}")
    print("  통과 조건:")
    for key, result in summary["conditions"].items():
        mark = "통과" if result["pass"] else "실패"
        extra = ""
        if key in ("1_false_alarm", "2_danger_detection"):
            extra = f"({result['exceed']}/{result['n']} = {_pct(result['rate'])})"
        if key == "2_danger_detection":
            extra += f", 위험 등급 판정 비율 {_pct(result['graded_danger_ratio'])}"
        if key == "3_grade_order":
            meds = result["grade_medians"]
            extra = " < ".join(f"{k} {_fmt(v)}" for k, v in meds.items())
            extra += f", 임계값 단조 {'예' if result['thresholds_monotonic'] else '아니오'}"
        if key in ("4_extent_order", "5_generalization"):
            fails = [c for c in result["checks"] if not c["ok"]]
            extra = f"실패 {len(fails)}/{len(result['checks'])}"
        print(f"   - {CONDITION_NAMES[key]}: {mark} {extra}")
    print(f"  판정: {summary['verdict']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--no-test", action="store_true", help="drop test split (development only)")
    args = parser.parse_args(argv)

    guide = loader.read_csv(args.data_dir / "column_guide.csv")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "no-test (development)" if args.no_test else "full",
        "prereg": {"path": str(PREREG.relative_to(ROOT)), "sha256": loader.file_digest(PREREG, "sha256")},
        "git": _git_state(),
        "inputs_md5": {
            p.name: loader.file_digest(p) for p in sorted(args.data_dir.glob("*")) if p.is_file()
        },
        "datasets": {},
    }
    with open(args.data_dir / "data_provenance.json", encoding="utf-8") as handle:
        report["model_version"] = json.load(handle).get("model_version")

    window_tables = []
    pairs = {ds: loader.load_pair(args.data_dir, ds, guide) for ds in loader.DATASETS}
    if args.no_test:
        pairs = {ds: (_drop_test(r), _drop_test(d)) for ds, (r, d) in pairs.items()}

    print(f"AI② 공개 데이터 등급 검증 · {report['model_version']} · 모드 {report['mode']}")
    print(f"사전 등록서 SHA-256 {report['prereg']['sha256'][:16]}… · 커밋 {report['git']['commit'][:8]}"
          f"{' (작업 중 변경 있음)' if report['git']['dirty_files'] else ''}")

    # Box
    raw, deg = pairs["ponte_moesa"]
    box = ev.prepare(deg, deg["tac_4_2_z"], "held_main")
    box_summary, box_windows = ev.evaluate_bridge(box, ev.BOX)
    box_summary["verdict"] = ev.verdict(box_summary["conditions"])
    normal_valid = box_summary["counts"]["DS2 test"]["valid"]
    box_summary["box_all_provisional"] = normal_valid < ev.MIN_WINDOWS
    if not args.no_test:
        box_summary["work_effect_raw"] = ev.box_work_effect(raw)
    box_summary["session_difference_raw"] = ev.box_session_difference(raw)
    report["datasets"]["ponte_moesa"] = {"indicator": "deg tac_4_2_z", **box_summary}
    window_tables.append(box_windows.assign(dataset="ponte_moesa", kind="deg", indicator="tac_4_2_z"))
    note = f"Box 등급 전체 잠정 (정상 test {normal_valid}개, 30개 미만)" if box_summary["box_all_provisional"] else None
    print_bridge("BOX · Ponte Moesa · 열화본 · 1-TAC 4-2 z", box_summary, note)
    if "work_effect_raw" in box_summary:
        w = box_summary["work_effect_raw"]
        pre = w["작업전"]["after_hold"]
        dur = w["작업중"]["after_hold"]
        print(f"  작업 영향(원본): 작업 전 {pre['exceed']}/{pre['n']}, 작업 중 {dur['exceed']}/{dur['n']} "
              f"({_pct(dur['rate'])}) → 한계 기록 {w['work_effect_limit']}")
    s = box_summary["session_difference_raw"]
    print(f"  세션 차이(원본 2-3 z): 저녁 {s['n_evening']}개 {_fmt(s['median_evening'])} / 낮 {s['n_day']}개 "
          f"{_fmt(s['median_day'])} / 차이 {_fmt(s['gap'])}")
    for stage, v in s["stages"].items():
        print(f"    {stage}: 변화 {_fmt(v['shift_from_day'])} → 잠정 {v['provisional']}")

    # Truss (HBTA)
    raw, deg = pairs["hbta"]
    values = ev.hbta_values(raw, deg)
    results = {}
    for kind, (frame, series, info) in values.items():
        prepared = ev.prepare(frame, series, "held_main")
        summary, windows = ev.evaluate_bridge(prepared, ev.TRUSS)
        summary["reference_info"] = info
        summary["verdict"] = ev.verdict(summary["conditions"])
        results[kind] = (summary, windows, prepared)
        window_tables.append(windows.assign(dataset="hbta", kind=kind, indicator="ar_distance"))
    deg_summary = results["deg_z"][0]
    deg_summary["cause_analysis"] = ev.hbta_cause(
        deg_summary["conditions"], results["raw_z"][0]["conditions"], results["raw_yz"][0]["conditions"])
    deg_summary["session_difference"] = ev.hbta_session(results["deg_z"][2])
    deg_summary["uds01_only_reference"] = ev.hbta_uds01_reference(results["deg_z"][2])
    deg_summary["conditions_by_stage"] = ev.hbta_conditions_table(deg)
    tac = ev.prepare(deg, deg["tac_a_b_z"], "held_main")
    tac_summary, tac_windows = ev.evaluate_bridge(tac, ev.TRUSS)
    tac_summary["verdict"] = ev.verdict(tac_summary["conditions"])
    window_tables.append(tac_windows.assign(dataset="hbta", kind="deg", indicator="tac_a_b_z"))
    report["datasets"]["hbta"] = {
        "indicator": "deg z AR(5) Mahalanobis",
        **deg_summary,
        "raw_z_reference": results["raw_z"][0],
        "raw_yz_reference": results["raw_yz"][0],
        "comparison_tac_a_b_z": tac_summary,
    }
    print_bridge("TRUSS · HBTA · 열화본 · AR(5) z 거리", deg_summary)
    if deg_summary["cause_analysis"]:
        print("  원인 분석:")
        for key, cause in deg_summary["cause_analysis"].items():
            print(f"   - {CONDITION_NAMES[key]}: {cause}")
    sd = deg_summary["session_difference"]
    print(f"  세션 차이: UDS_01 {_fmt(sd['UDS_01']['median'])} / UDS_02 {_fmt(sd['UDS_02']['median'])} / 차이 {_fmt(sd['gap'])}")
    print_bridge("TRUSS · HBTA · 열화본 · 비교 지표 1-TAC z (참고)", tac_summary)

    # Old ADA
    raw, deg = pairs["old_ada"]
    report["datasets"]["old_ada"] = {kind: ev.old_ada_report(f, kind) for kind, f in (("raw", raw), ("deg", deg))}
    print("\n[TRUSS · Old ADA · 점검용 (통과 판정 없음)]")
    for kind in ("raw", "deg"):
        for name, ind in report["datasets"]["old_ada"][kind]["indicators"].items():
            print(f"  {kind} {name}: 참고선 {_fmt(ind['reference_line_ai2_calib_max'])}, "
                  f"정상 test 초과 {ind['normal_test_above_line']}/{ind['normal_test_n']}, "
                  f"test 중앙값 순서(낮음→높음) {' < '.join(ind['test_median_order_low_to_high'])}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    args.windows.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(window_tables, ignore_index=True).to_csv(args.windows, index=False, encoding="utf-8-sig")
    print(f"\n요약: {args.summary}\nWindow별 등급: {args.windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
