#!/usr/bin/env bash
# D1 야간 GPU 대기열: p2_coco CV 종료를 기다린 뒤 순서대로 실행
cd "$(dirname "$0")/.."
PY=${PY:-python}
while pgrep -f "src.models.cv --tag p2_coco" > /dev/null; do sleep 30; done
$PY -m src.models.oof_yolo --tag p2_coco > experiments/logs/oof_p2_coco.log 2>&1
$PY -m src.models.cv --tag p2_scratch --scratch > experiments/logs/cv_p2_scratch.log 2>&1
$PY -m src.models.oof_yolo --tag p2_scratch > experiments/logs/oof_p2_scratch.log 2>&1
$PY -m src.models.cv --tag p3_coco --arch yolov8s.yaml > experiments/logs/cv_p3_coco.log 2>&1
$PY -m src.models.oof_yolo --tag p3_coco > experiments/logs/oof_p3_coco.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/raw_f3.yaml --name raw_p2_coco_f3 --epochs 50 > experiments/logs/raw_f3.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/dev_all.yaml --name p2_coco_devall --epochs 50 > experiments/logs/devall_p2_coco.log 2>&1
echo QUEUE_DONE > experiments/logs/gpu_queue_d1.done
