#!/usr/bin/env bash
# 주모델 v2(p2_coco_v2) OOF가 생기면 CPU에서 분석 일괄 실행
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
while [ ! -f artifacts/preds/p2_coco_v2/oof_info.json ]; do sleep 30; done
$PY -m src.eval.compare p2_coco_v2 > experiments/logs/compare_v2.log 2>&1
$PY -m src.analysis.thresholds --tag p2_coco_v2 > experiments/logs/thresholds_v2.log 2>&1
$PY -m src.analysis.errors --tag p2_coco_v2 > experiments/logs/errors_v2.log 2>&1
$PY -m src.analysis.leak_tests --tag p2_coco_v2 > experiments/logs/leak_tests_v2.log 2>&1
$PY -m src.analysis.stress --tag p2_coco_v2 > experiments/logs/stress_v2.log 2>&1
$PY -m src.analysis.uncertainty --tag p2_coco_v2 > experiments/logs/uncertainty_v2.log 2>&1
$PY -m src.report.figures_report > experiments/logs/figures_report.log 2>&1
echo DONE > experiments/logs/cpu_queue_v2.done
