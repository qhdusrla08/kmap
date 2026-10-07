#!/usr/bin/env bash
# D2 GPU 대기열: D1 대기열 종료 후 주모델 v2(실제 흔적-무이물 음성 추가) CV·최종 학습·일반화 평가 (dev 안에서만)
cd "$(dirname "$0")/.."
PY=${PY:-python}
while [ ! -f experiments/logs/gpu_queue_d1.done ]; do sleep 60; done
$PY -m src.models.cv --tag p2_coco_v2 --data_sfx _v2 > experiments/logs/cv_p2_coco_v2.log 2>&1
$PY -m src.models.oof_yolo --tag p2_coco_v2 > experiments/logs/oof_p2_coco_v2.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/dev_all_v2.yaml --name p2_coco_v2_devall --epochs 50 > experiments/logs/devall_p2_coco_v2.log 2>&1
for s in lomo_m1 lomo_m2 lomo_m3 forward; do
  $PY -m src.models.holdout --split ${s}_v2 --tag p2_coco_v2 > experiments/logs/holdout_${s}_v2.log 2>&1
done
echo QUEUE_DONE > experiments/logs/gpu_queue_d2.done
