"""전체 파이프라인 단일 실행: 인덱싱 → 진단 → 전처리 → 분할 → 학습 → 추론 → 평가 → 결과 생성.

사용:
  python run_all.py                    # full: 비교 모델·ablation·일반화 평가 포함 (RTX 3070 기준 약 8–9시간, 단계별 실측 시간 합산 추정)
  python run_all.py --profile core     # core: 베이스라인 + 최종모델 + 최종 평가 (약 1시간 10분, 깨끗한 디렉터리 실측 70.5분)
  python run_all.py --only index       # 특정 단계만
  python run_all.py --from thresholds  # 해당 단계부터
  python run_all.py --list             # 단계 목록
환경변수 KMAP_DATA_ROOT로 원본 데이터 위치를 바꿀 수 있습니다 (기본: ./data).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = "p3_coco_v2b"  # 최종모델: YOLOv8s-P3 + v2b 학습 사본 (선정 근거: docs/decisions.md 2026-10-03)
P3 = ["--arch", "yolov8s.yaml"]

# (이름, 모듈, 인자, 프로필) — core 단계는 full에도 포함됩니다.
STAGES = [
    # 데이터·전처리
    ("index", "src.data.index", [], "core"),
    ("figures_data", "src.analysis.figures_data", [], "core"),
    ("qc_marking", "src.analysis.qc_marking", [], "full"),
    ("preprocess", "src.data.preprocess", [], "core"),
    ("split", "src.data.split", [], "core"),
    ("yolo_export", "src.data.yolo_export", [], "core"),
    ("export_v2b", "src.data.export_v2", ["--name", "v2b", "--erase", "coremask"], "core"),
    ("figures_prep", "src.report.figures_prep", [], "core"),  # 제1장 그림(시각·제품 유형, 표시 제거, 가짜 표시, 이식·흔적 음성)
    ("export_v2", "src.data.export_v2", ["--name", "v2", "--erase", "core5"], "full"),
    ("export_v2bn", "src.data.export_noise", [], "full"),  # 3호기 대책 비교용 잡음 정합 사본 (ablation)
    ("raw_export", "src.data.raw_export", [], "full"),
    ("leak_t1", "src.analysis.leak_t1", [], "full"),
    ("conditions", "src.analysis.conditions", [], "core"),
    # 베이스라인
    ("hgb_oof", "src.models.baseline_hgb", ["--mode", "oof"], "core"),
    ("hgb_v2_oof", "src.models.baseline_hgb", ["--mode", "oof", "--variant", "v2"], "full"),
    # 최종모델 CV
    ("cv_main", "src.models.cv", ["--tag", MAIN, *P3, "--data_sfx", "_v2b"], "core"),
    ("oof_main", "src.models.oof_yolo", ["--tag", MAIN], "core"),
    # 비교 모델 (같은 분할·같은 평가)
    ("cv_p2_v1", "src.models.cv", ["--tag", "p2_coco"], "full"),
    ("oof_p2_v1", "src.models.oof_yolo", ["--tag", "p2_coco"], "full"),
    ("cv_p2_scratch", "src.models.cv", ["--tag", "p2_scratch", "--scratch"], "full"),
    ("oof_p2_scratch", "src.models.oof_yolo", ["--tag", "p2_scratch"], "full"),
    ("cv_p3_v1", "src.models.cv", ["--tag", "p3_coco", *P3], "full"),
    ("oof_p3_v1", "src.models.oof_yolo", ["--tag", "p3_coco"], "full"),
    ("cv_p2_v2b", "src.models.cv", ["--tag", "p2_coco_v2b", "--data_sfx", "_v2b"], "full"),
    ("oof_p2_v2b", "src.models.oof_yolo", ["--tag", "p2_coco_v2b"], "full"),
    ("cv_p3_v2bn", "src.models.cv", ["--tag", "p3_coco_v2bn", *P3, "--data_sfx", "_v2bn"], "full"),
    ("oof_p3_v2bn", "src.models.oof_yolo", ["--tag", "p3_coco_v2bn"], "full"),
    ("nested_main", "src.models.nested", ["--tag", MAIN, *P3, "--data_sfx", "_v2b", "--outer", "2,3"], "full"),
    ("raw_f3", "src.models.yolo", ["--data", "artifacts/yolo/raw_f3.yaml", "--name", "raw_p2_coco_f3"], "full"),
    ("leak_t2", "src.analysis.leak_t2", [], "full"),
    ("compare_core", "src.eval.compare", ["hgb", MAIN], "core"),
    ("compare", "src.eval.compare", ["hgb", "hgb_v2", "p2_coco", "p2_scratch", "p3_coco", "p2_coco_v2b", "p3_coco_v2bn"], "full"),
    # 보정·3구간·오류·지름길·스트레스·불확실성
    ("thresholds_main", "src.analysis.thresholds", ["--tag", MAIN, "--rule", "margin", "--ood_guard"], "core"),  # 규칙 근거: threshold_transfer, docs/decisions.md
    ("thresholds_hgb", "src.analysis.thresholds", ["--tag", "hgb"], "core"),
    ("threshold_transfer_lofo", "src.analysis.threshold_transfer", ["--tag", MAIN, "--modes", "lofo"], "core"),
    ("threshold_transfer", "src.analysis.threshold_transfer", ["--tag", MAIN, "--modes", "lofo,nested"], "full"),
    ("errors_main", "src.analysis.errors", ["--tag", MAIN], "core"),
    ("leak_tests_main", "src.analysis.leak_tests", ["--tag", MAIN], "core"),
    ("stress_main", "src.analysis.stress", ["--tag", MAIN], "core"),
    ("uncertainty_main", "src.analysis.uncertainty", ["--tag", MAIN], "core"),
    ("thresholds_cmp", "src.analysis.thresholds", ["--tag", "p2_coco"], "full"),
    ("leak_tests_hgb", "src.analysis.leak_tests", ["--tag", "hgb"], "full"),
    ("leak_tests_p2_v1", "src.analysis.leak_tests", ["--tag", "p2_coco"], "full"),
    ("leak_tests_p2_v2b", "src.analysis.leak_tests", ["--tag", "p2_coco_v2b"], "full"),
    ("stress_hgb", "src.analysis.stress", ["--tag", "hgb"], "full"),
    ("stress_hgb_v2", "src.analysis.stress", ["--tag", "hgb_v2"], "full"),
    ("stress_p2_v1", "src.analysis.stress", ["--tag", "p2_coco"], "full"),
    ("stress_p2_v2b", "src.analysis.stress", ["--tag", "p2_coco_v2b"], "full"),
    ("ablation_pre", "src.analysis.ablation_pre", [], "full"),
    ("thresholds_v2bn", "src.analysis.thresholds", ["--tag", "p3_coco_v2bn"], "full"),
    ("threshold_transfer_v2bn", "src.analysis.threshold_transfer", ["--tag", "p3_coco_v2bn", "--modes", "lofo"], "full"),
    ("leak_tests_v2bn", "src.analysis.leak_tests", ["--tag", "p3_coco_v2bn"], "full"),
    # 일반화 (dev 안)
    ("holdout_lomo_m1", "src.models.holdout", ["--split", "lomo_m1_v2b", "--tag", MAIN, *P3], "full"),
    ("holdout_lomo_m2", "src.models.holdout", ["--split", "lomo_m2_v2b", "--tag", MAIN, *P3], "full"),
    ("holdout_lomo_m3", "src.models.holdout", ["--split", "lomo_m3_v2b", "--tag", MAIN, *P3], "full"),
    ("holdout_forward", "src.models.holdout", ["--split", "forward_v2b", "--tag", MAIN, *P3], "full"),
    ("holdout_lomo_m3_v2bn", "src.models.holdout", ["--split", "lomo_m3_v2bn", "--tag", "p3_coco_v2bn", *P3], "full"),
    ("generalization", "src.eval.generalization", ["--tag", MAIN], "full"),
    ("noise_ablation", "src.analysis.noise_ablation", [], "full"),
    # 최종 학습 → 동결 테스트 1회 평가 → 예측 파일
    ("devall_main", "src.models.yolo", ["--data", "artifacts/yolo/dev_all_v2b.yaml", "--name", f"{MAIN}_devall", *P3], "core"),
    ("hgb_final", "src.models.baseline_hgb", ["--mode", "final"], "core"),
    ("final", "src.eval.final", ["--tag", MAIN], "core"),
    ("speed", "src.eval.speed", ["--tag", MAIN], "core"),
    ("deploy_shift", "src.analysis.deploy_shift", ["--tag", MAIN], "full"),
    ("monitoring", "src.analysis.monitoring", [], "core"),
    ("quality_drift", "src.analysis.quality_drift", [], "full"),
    ("audit_pass", "src.analysis.audit_pass", [], "full"),
    ("figures_report", "src.report.figures_report", [], "core"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["core", "full"], default="full")
    ap.add_argument("--only", nargs="*", help="실행할 단계 이름")
    ap.add_argument("--from", dest="start", help="이 단계부터 실행")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    stages = [s for s in STAGES if a.profile == "full" or s[3] == "core"]
    names = [s[0] for s in stages]
    if a.list:
        for n, m, args, prof in stages:
            print(f"{n:18s} [{prof}] python -m {m} {' '.join(args)}")
        return 0
    todo = stages
    if a.only:
        bad = set(a.only) - set(names)
        if bad:
            ap.error(f"unknown stages: {sorted(bad)}")
        todo = [s for s in stages if s[0] in a.only]
    elif a.start:
        if a.start not in names:
            ap.error(f"unknown stage: {a.start}")
        todo = stages[names.index(a.start):]

    log_dir = ROOT / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    status = []
    for name, module, args, _ in todo:
        print(f"RUN  {name}", flush=True)
        t0 = time.monotonic()
        with open(log_dir / f"{name}.log", "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = round(time.monotonic() - t0, 1)
        status.append(dict(stage=name, exit_code=r.returncode, seconds=dt))
        (log_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if r.returncode:
            print(f"FAIL {name} ({dt}s) → experiments/logs/{name}.log", flush=True)
            return r.returncode
        print(f"PASS {name} ({dt}s)", flush=True)
    print("All stages completed. Outputs: outputs/, logs: experiments/logs/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
