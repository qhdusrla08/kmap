"""단일 hold-out 학습 + 그 검증 목록 추론 (LOMO, 전향 평가).
사용: python -m src.models.holdout --split lomo_m1 [--tag p2_coco]"""
import argparse
import json
import re

import yaml

from src.config import ROOT, load_config
from src.models.predict import yolo_predict
from src.models.yolo import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--arch", default="yolov8s-p2.yaml")
    ap.add_argument("--scratch", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    name = f"{a.tag}_{a.split}"
    yml = ROOT / "artifacts" / "yolo" / f"{a.split}.yaml"
    run = train(yml, name, arch=a.arch, pretrained=not a.scratch, seed=cfg["seed"])
    val = yaml.safe_load(yml.read_text())["val"]  # 검증 목록은 yaml에서 읽음 (v2 분할도 같은 clean 검증 목록 사용)
    pr, sp = yolo_predict(run / "weights" / "last.pt", [l for l in open(val).read().splitlines() if l])
    out = ROOT / "artifacts" / "preds" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    pr.to_csv(out / f"{re.sub(r'_v2bn?$|_v2$', '', a.split)}.csv", index=False)  # lomo_m1_v2b → lomo_m1.csv
    print(json.dumps(dict(split=a.split, n_pred=len(pr), **sp)))


if __name__ == "__main__":
    main()
