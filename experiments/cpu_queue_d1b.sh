#!/usr/bin/env bash
# 주모델 CPU 분석이 끝나면: HGB v2 누수·3구간·오류 → 전처리 ablation 재실행(같은 삭제 방식)
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
while [ ! -f experiments/logs/cpu_queue_main.done ]; do sleep 30; done
$PY -m src.analysis.thresholds --tag hgb_v2 > experiments/logs/thresholds_hgb_v2.log 2>&1
$PY -m src.analysis.errors --tag hgb_v2 > experiments/logs/errors_hgb_v2.log 2>&1
$PY -m src.analysis.leak_tests --tag hgb_v2 > experiments/logs/leak_tests_hgb_v2.log 2>&1
$PY -m src.analysis.ablation_pre > experiments/logs/ablation_pre.log 2>&1
echo DONE > experiments/logs/cpu_queue_d1b.done
