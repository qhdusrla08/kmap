"""fold별 YOLO(last.pt)로 해당 검증 fold를 추론해 OOF 예측을 만듭니다.
사용: python -m src.models.oof_yolo --tag p2_coco"""
import argparse
import json

import pandas as pd

from src.config import ROOT
from src.models.predict import read_list, yolo_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", default="last.pt")
    a = ap.parse_args()
    out = ROOT / "artifacts" / "preds" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    parts, info = [], {}
    for f in range(4):
        w = ROOT / "artifacts" / "runs" / f"{a.tag}_f{f}" / "weights" / a.weights
        pr, sp = yolo_predict(w, read_list(f"f{f}_val"))
        parts.append(pr.assign(fold=f))
        info[f"f{f}"] = sp
        print(f"fold {f}", json.dumps(sp), flush=True)
    pd.concat(parts).to_csv(out / "oof.csv", index=False)
    (out / "oof_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
