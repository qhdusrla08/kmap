#!/usr/bin/env bash
# P3 v2 OOF가 생기면 CPU 분석 (비교·3구간·오류·누수·스트레스)
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
while [ ! -f artifacts/preds/p3_coco_v2/oof_info.json ]; do sleep 30; done
$PY -m src.eval.compare p3_coco_v2 > experiments/logs/compare_p3v2.log 2>&1
$PY -m src.analysis.thresholds --tag p3_coco_v2 > experiments/logs/thresholds_p3v2.log 2>&1
$PY -m src.analysis.errors --tag p3_coco_v2 > experiments/logs/errors_p3v2.log 2>&1
$PY -m src.analysis.leak_tests --tag p3_coco_v2 > experiments/logs/leak_tests_p3v2.log 2>&1
$PY -m src.analysis.stress --tag p3_coco_v2 > experiments/logs/stress_p3v2.log 2>&1
echo DONE > experiments/logs/cpu_queue_p3v2.done
