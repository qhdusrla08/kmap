#!/usr/bin/env bash
# D5 보완 실험 GPU 대기열 (2026-10-03): A0 배포 이동 → A2b 중첩 학습 → B3 잡음 정합 증강 dev 비교
# 사용: bash experiments/gpu_queue_d5.sh [기다릴 PID]
cd "$(dirname "$0")/.."
PY=${PY:-python}
L=experiments/logs
if [ -n "$1" ]; then while kill -0 "$1" 2>/dev/null; do sleep 30; done; fi
date '+start %F %T' > $L/d5_queue.status
$PY -m src.analysis.deploy_shift --tag p3_coco_v2b > $L/deploy_shift.log 2>&1; echo "A0 $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.models.nested --tag p3_coco_v2b --arch yolov8s.yaml --data_sfx _v2b --outer 2,3 > $L/nested.log 2>&1; echo "A2b $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.models.cv --tag p3_coco_v2bn --arch yolov8s.yaml --data_sfx _v2bn > $L/cv_p3_v2bn.log 2>&1; echo "B3cv $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.models.oof_yolo --tag p3_coco_v2bn > $L/oof_p3_v2bn.log 2>&1; echo "B3oof $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.models.holdout --split lomo_m3_v2bn --tag p3_coco_v2bn --arch yolov8s.yaml > $L/holdout_lomo_m3_p3v2bn.log 2>&1; echo "B3lomo $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.eval.compare p3_coco_v2bn > $L/compare_v2bn.log 2>&1; echo "B3cmp $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.analysis.thresholds --tag p3_coco_v2bn > $L/thresholds_v2bn.log 2>&1; echo "B3thr $? $(date +%T)" >> $L/d5_queue.status
$PY -m src.analysis.leak_tests --tag p3_coco_v2bn > $L/leak_tests_v2bn.log 2>&1; echo "B3leak $? $(date +%T)" >> $L/d5_queue.status
echo DONE > $L/d5_queue.done
