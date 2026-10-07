#!/usr/bin/env bash
# D1 GPU 대기열 종료 후: P3 비교표 추가, T2 누수 상한(raw 학습 모델) 분석
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KMAP_DEVICE=cpu OMP_NUM_THREADS=4
while [ ! -f experiments/logs/gpu_queue_d1.done ]; do sleep 30; done
$PY -m src.eval.compare p3_coco > experiments/logs/compare_p3.log 2>&1
$PY -m src.analysis.leak_t2 > experiments/logs/leak_t2.log 2>&1
echo DONE > experiments/logs/cpu_queue_d1c.done
