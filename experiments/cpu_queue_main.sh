#!/usr/bin/env bash
# 주모델(p2_coco) OOF가 생기면 CPU에서 분석 일괄 실행 (GPU는 다음 학습이 사용)
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
while [ ! -f artifacts/preds/p2_coco/oof_info.json ]; do sleep 20; done
$PY -m src.eval.compare hgb p2_coco > experiments/logs/compare_core.log 2>&1
$PY -m src.analysis.thresholds --tag p2_coco > experiments/logs/thresholds_main.log 2>&1
$PY -m src.analysis.thresholds --tag hgb > experiments/logs/thresholds_hgb.log 2>&1
$PY -m src.analysis.errors --tag p2_coco > experiments/logs/errors_main.log 2>&1
$PY -m src.analysis.errors --tag hgb > experiments/logs/errors_hgb.log 2>&1
$PY -m src.analysis.leak_tests --tag p2_coco > experiments/logs/leak_tests_main.log 2>&1
$PY -m src.analysis.stress --tag p2_coco > experiments/logs/stress_main.log 2>&1
$PY -m src.analysis.uncertainty --tag p2_coco > experiments/logs/uncertainty_main.log 2>&1
echo DONE > experiments/logs/cpu_queue_main.done
