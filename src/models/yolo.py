"""Ultralytics YOLOv8 학습 래퍼 (P2 소형객체 헤드).

외부 자원 통제 (docs/external_data.md):
- COCO 사전학습 yolov8s.pt만 명시적으로 사용합니다(--scratch면 사용 안 함).
- AMP 점검용 자동 다운로드(yolo11n.pt)를 막기 위해 check_amp를 고정 True로 대체합니다(RTX 30xx는 AMP 지원).
- 학습 중 그림 생성(plots)을 끄고 폰트 다운로드를 피합니다. 사용 통계 전송(sync)을 끕니다.
"""
import argparse
import json
from pathlib import Path

from src.config import ROOT, load_config

AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=0.0, translate=0.1, scale=0.2, shear=0.0, perspective=0.0,
           flipud=0.5, fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=10)


def setup_ultralytics():
    from ultralytics import settings
    settings.update({"sync": False})
    import ultralytics.engine.trainer as tr
    tr.check_amp = lambda model: True


def build_model(arch="yolov8s-p2.yaml", pretrained=True):
    from ultralytics import YOLO
    m = YOLO(arch)
    if pretrained:
        w = ROOT / "artifacts" / "weights" / "yolov8s.pt"
        if not w.exists():
            w.parent.mkdir(parents=True, exist_ok=True)
            from ultralytics.utils.downloads import attempt_download_asset
            src = attempt_download_asset("yolov8s.pt")
            Path(src).replace(w)
        m.load(str(w))
    return m


def train(data, name, epochs=50, imgsz=640, batch=16, arch="yolov8s-p2.yaml", pretrained=True, seed=42,
          fraction=1.0, patience=0, workers=6):
    # patience=0: early stopping 끔. 검증 fold로 학습 길이를 고르지 않아 OOF 선택 편향을 막습니다.
    setup_ultralytics()
    m = build_model(arch, pretrained)
    project = ROOT / "artifacts" / "runs"
    res = m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, deterministic=True,
                  project=str(project), name=name, exist_ok=True, pretrained=pretrained, optimizer="AdamW", lr0=0.001,
                  cos_lr=True, patience=patience, workers=workers, cache=False, plots=False, fraction=fraction,
                  single_cls=True, max_det=20, verbose=False, amp=True, val=True, **AUG)
    return project / name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset yaml (artifacts/yolo/f0.yaml 등)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--arch", default="yolov8s-p2.yaml")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0)
    a = ap.parse_args()
    cfg = load_config()
    out = train(a.data, a.name, a.epochs, a.imgsz, a.batch, a.arch, not a.scratch, cfg["seed"], a.fraction)
    print(json.dumps({"run": str(out.relative_to(ROOT))}))
