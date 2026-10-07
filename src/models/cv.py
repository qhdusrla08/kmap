"""fold별 YOLO 학습 (OOF용). 사용: python -m src.models.cv --tag p2_coco [--scratch] [--arch yolov8s.yaml]"""
import argparse
import json

from src.config import ROOT, load_config
from src.models.yolo import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="yolov8s-p2.yaml")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--data_sfx", default="", help="'_v2'이면 f{j}_v2.yaml(실제 흔적-무이물 음성 추가 사본) 사용")
    a = ap.parse_args()
    cfg = load_config()
    for f in [int(x) for x in a.folds.split(",")]:
        run = train(ROOT / "artifacts" / "yolo" / f"f{f}{a.data_sfx}.yaml", f"{a.tag}_f{f}", epochs=a.epochs, arch=a.arch,
                    pretrained=not a.scratch, seed=cfg["seed"])
        print(json.dumps({"fold": f, "run": str(run.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
