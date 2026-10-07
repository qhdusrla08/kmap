#!/usr/bin/env bash
# v2b OOF가 생기면 CPU 분석 (P2 → P3 순)
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
for tag in p2_coco_v2b p3_coco_v2b; do
  while [ ! -f artifacts/preds/$tag/oof_info.json ]; do sleep 30; done
  $PY -m src.eval.compare $tag > experiments/logs/compare_$tag.log 2>&1
  $PY -m src.analysis.thresholds --tag $tag > experiments/logs/thresholds_$tag.log 2>&1
  $PY -m src.analysis.errors --tag $tag > experiments/logs/errors_$tag.log 2>&1
  $PY -m src.analysis.leak_tests --tag $tag > experiments/logs/leak_tests_$tag.log 2>&1
  $PY -m src.analysis.stress --tag $tag > experiments/logs/stress_$tag.log 2>&1
  $PY -m src.analysis.uncertainty --tag $tag > experiments/logs/uncertainty_$tag.log 2>&1
done
echo DONE > experiments/logs/cpu_queue_v2b.done
