"""YOLO 추론 → 원본 픽셀 좌표 예측 표. 입력은 clean(마킹 제거) PNG만 허용합니다."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT


def check_inputs(paths, allow_raw=False):
    ok = ("/images/clean/", "/images/eval_") + (("/images/raw/",) if allow_raw else ())
    bad = [p for p in paths if not any(k in str(p) for k in ok)]
    if bad:
        raise ValueError(f"추론 입력은 마킹 제거 영상이어야 합니다: {bad[:3]}")


def yolo_predict(weights, paths, conf=0.001, iou=0.6, imgsz=640, max_det=20, augment=False, batch=16, allow_raw=False):
    from ultralytics import YOLO
    from src.models.yolo import setup_ultralytics
    setup_ultralytics()
    paths = [str(p) for p in paths]
    check_inputs(paths, allow_raw)  # raw(마킹 잔존)는 누수 진단(T2)에서만 명시적으로 허용
    model = YOLO(str(weights))
    rows, times = [], []
    for k in range(0, len(paths), batch):
        chunk = paths[k:k + batch]
        res = model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, augment=augment,
                            verbose=False, half=False, batch=len(chunk), device=os.environ.get("KMAP_DEVICE"))
        for p, r in zip(chunk, res):
            iid = Path(p).stem
            times.append(sum(r.speed.values()))
            b = r.boxes
            if b is None or not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy()
            sc = b.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, sc):
                rows.append((iid, float(x1), float(y1), float(x2), float(y2), float(s)))
    df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"])
    return df, dict(ms_per_image_mean=float(np.mean(times)), ms_per_image_p95=float(np.percentile(times, 95)))


def read_list(name):
    p = ROOT / "artifacts" / "yolo" / "lists" / f"{name}.txt"
    return [l for l in p.read_text().splitlines() if l]


def predict_fold(tag, fold, paths, weights="last.pt"):
    """fold 모델로 추론 (YOLO 태그 또는 'hgb'). 누수·스트레스 시험에서 모델 간 동일 인터페이스로 사용."""
    paths = [str(p) for p in paths]
    check_inputs(paths)
    if tag.startswith("hgb"):
        import joblib
        from src.models import baseline_hgb as hb
        m = joblib.load(ROOT / "artifacts" / "preds" / tag / f"model_f{fold}.joblib")
        return hb.predict(m, paths)
    return yolo_predict(ROOT / "artifacts" / "runs" / f"{tag}_f{fold}" / "weights" / weights, paths)
