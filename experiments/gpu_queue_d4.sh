#!/usr/bin/env bash
# v2b(코어 마스크 삭제) 재학습: P2·P3 CV → OOF → dev 전체 학습. v2(5×5 삭제)는 잔존 신호 결함으로 폐기
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -m src.models.cv --tag p2_coco_v2b --data_sfx _v2b > experiments/logs/cv_p2_coco_v2b.log 2>&1
$PY -m src.models.oof_yolo --tag p2_coco_v2b > experiments/logs/oof_p2_coco_v2b.log 2>&1
$PY -m src.models.cv --tag p3_coco_v2b --arch yolov8s.yaml --data_sfx _v2b > experiments/logs/cv_p3_coco_v2b.log 2>&1
$PY -m src.models.oof_yolo --tag p3_coco_v2b > experiments/logs/oof_p3_coco_v2b.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/dev_all_v2b.yaml --name p2_coco_v2b_devall --epochs 50 > experiments/logs/devall_p2_coco_v2b.log 2>&1
$PY -m src.models.yolo --data artifacts/yolo/dev_all_v2b.yaml --name p3_coco_v2b_devall --arch yolov8s.yaml --epochs 50 > experiments/logs/devall_p3_coco_v2b.log 2>&1
echo QUEUE_DONE > experiments/logs/gpu_queue_d4.done
