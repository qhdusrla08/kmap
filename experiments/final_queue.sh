#!/usr/bin/env bash
# 4단계: 최종모델(p3_coco_v2b) 호기 hold-out·전향 → 5단계: 동결 테스트 1회 평가, 속도, 모니터링
cd "$(dirname "$0")/.."
PY=${PY:-python}
for s in lomo_m1 lomo_m2 lomo_m3 forward; do
  $PY -m src.models.holdout --split ${s}_v2b --tag p3_coco_v2b --arch yolov8s.yaml > experiments/logs/holdout_${s}_p3v2b.log 2>&1
done
echo STEP4_TRAIN_DONE > experiments/logs/final_step4.done
while [ ! -f artifacts/preds/hgb/model_dev_all.joblib ] || pgrep -f "baseline_hgb --mode final" > /dev/null; do sleep 20; done
$PY -m src.eval.final --tag p3_coco_v2b > experiments/logs/final.log 2>&1
$PY -m src.eval.speed --tag p3_coco_v2b > experiments/logs/speed.log 2>&1
$PY -m src.analysis.monitoring > experiments/logs/monitoring.log 2>&1
echo STEP5_DONE > experiments/logs/final_step5.done
OMP_NUM_THREADS=4 $PY -m src.eval.generalization --tag p3_coco_v2b > experiments/logs/generalization.log 2>&1
echo DONE > experiments/logs/final_queue.done
