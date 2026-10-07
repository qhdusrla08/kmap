"""영상당 처리 시간 측정 (batch 1, 실시간 적용 기준). 출력: outputs/tables/speed.json

- 전처리: 팔레트 BMP 읽기 + 마킹 제거(2px, telea_n)
- YOLO 주모델: GPU / CPU, 640 입력, 예열 10장 후 측정
- HGB 베이스라인: CPU
- TTA 추가 시간(GPU): 현행 판정 규칙은 통과 후보 영상(영상 점수 < t_low)에 반전 3종(좌우·상하·둘 다)을 더 추론해
  점수 표준편차를 봅니다. 영상을 메모리에서 뒤집고 3회 추론하는 시간을 batch 1로 잽니다.
  --tta_only: 기존 speed.json의 다른 측정값은 그대로 두고 이 항목만 추가합니다.
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import marking as mk
from src.models.predict import read_list


def tta_extra_ms(tag, paths):
    """통과 후보 영상 1장에 대한 TTA 추가 시간: 반전 3종을 만들고 각각 추론(GPU, batch 1, 예열 10장)."""
    import cv2
    import torch
    from ultralytics import YOLO
    from src.models.yolo import setup_ultralytics
    setup_ultralytics()
    m = YOLO(str(ROOT / "artifacts" / "runs" / f"{tag}_devall" / "weights" / "last.pt"))
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    run = lambda im: [m.predict(cv2.flip(im, c), imgsz=640, conf=0.001, device="cuda:0", verbose=False) for c in (1, 0, -1)]
    for im in imgs[:10]:
        run(im)
    tt = []
    for im in imgs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(im)
        torch.cuda.synchronize()
        tt.append((time.perf_counter() - t0) * 1000)
    return dict(mean=float(np.mean(tt)), p95=float(np.percentile(tt, 95)), views=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tta_only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.tta_only:
        out = cfg["paths"]["tables"] / "speed.json"
        res = json.loads(out.read_text())
        res["tta_extra_cuda_ms"] = tta_extra_ms(a.tag, read_list("test_clean")[: a.n])
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    files = idx[idx.is_canonical].file.iloc[: a.n].tolist()
    res = {"n_images": a.n, "cpu": platform.processor() or platform.machine()}
    t = []
    for f in files:
        t0 = time.perf_counter()
        img, _ = mk.remove_marking(mk.read_index(cfg["data_root"] / f), cfg["marking"]["dilate_px"], cfg["marking"]["inpaint"])
        t.append((time.perf_counter() - t0) * 1000)
    res["preprocess_ms"] = dict(mean=float(np.mean(t)), p95=float(np.percentile(t, 95)))
    import torch
    from ultralytics import YOLO
    from src.models.yolo import setup_ultralytics
    setup_ultralytics()
    paths = read_list("test_clean")[: a.n]
    w = ROOT / "artifacts" / "runs" / f"{a.tag}_devall" / "weights" / "last.pt"
    for dev in (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]:
        m = YOLO(str(w))
        for p in paths[:10]:
            m.predict(p, imgsz=640, conf=0.001, device=dev, verbose=False)
        tt = []
        for p in paths:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m.predict(p, imgsz=640, conf=0.001, device=dev, verbose=False)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            tt.append((time.perf_counter() - t0) * 1000)
        res[f"yolo_{dev.split(':')[0]}_ms"] = dict(mean=float(np.mean(tt)), p95=float(np.percentile(tt, 95)))
        if dev.startswith("cuda"):
            res["gpu"] = torch.cuda.get_device_name(0)
    import joblib
    from src.models import baseline_hgb as hb
    hm = joblib.load(ROOT / "artifacts" / "preds" / "hgb" / "model_dev_all.joblib")
    _, sp = hb.predict(hm, paths)
    res["hgb_cpu_ms"] = sp
    if torch.cuda.is_available():
        res["tta_extra_cuda_ms"] = tta_extra_ms(a.tag, paths)
    (cfg["paths"]["tables"] / "speed.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
