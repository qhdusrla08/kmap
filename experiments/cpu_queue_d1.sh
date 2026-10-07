#!/usr/bin/env bash
# D1 CPU 대기열: HGB 누수·스트레스 시험이 끝나면 전처리 ablation 실행
cd "$(dirname "$0")/.."
PY=${PY:-python}
while pgrep -f "src.analysis.(leak_tests|stress) --tag hgb" > /dev/null; do sleep 20; done
OMP_NUM_THREADS=4 $PY -m src.analysis.ablation_pre > experiments/logs/ablation_pre.log 2>&1
echo DONE > experiments/logs/cpu_queue_d1.done
