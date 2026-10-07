#!/usr/bin/env bash
# D3 GPU 대기열: P3 헤드에도 v2 사본을 적용해 P2 v2와 같은 조건으로 비교 (사전 기준 3순위 AP50:95에서 P3 v1이 앞섬)
cd "$(dirname "$0")/.."
PY=${PY:-python}
while [ ! -f experiments/logs/gpu_queue_d2.done ]; do sleep 60; done
$PY -m src.models.cv --tag p3_coco_v2 --arch yolov8s.yaml --data_sfx _v2 > experiments/logs/cv_p3_coco_v2.log 2>&1
$PY -m src.models.oof_yolo --tag p3_coco_v2 > experiments/logs/oof_p3_coco_v2.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/dev_all_v2.yaml --name p3_coco_v2_devall --arch yolov8s.yaml --epochs 50 > experiments/logs/devall_p3_coco_v2.log 2>&1
echo QUEUE_DONE > experiments/logs/gpu_queue_d3.done
