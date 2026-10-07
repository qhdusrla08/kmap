"""중첩 시뮬레이션용 안쪽 모델 (3구간 임계값 이식 검증, dev 전용).

실제 절차는 'fold 모델(dev 3/4로 학습)의 OOF로 보정·임계값을 정하고 → dev 전체로 재학습한 모델에 적용'입니다.
이를 dev 안에서 그대로 재현합니다.
  - 바깥 fold k의 배포 모델 = 기존 fold 모델 {tag}_f{k} (나머지 3개 fold로 학습)
  - 안쪽 모델 {tag}_nest_k{k}_j{j} = 바깥 k와 안쪽 j를 뺀 2개 fold로 학습 → fold j 추론(안쪽 OOF)
  - 안쪽 모델로 바깥 fold k도 추론해 같은 객체에서 '배포 − 안쪽' 점수 차이(배포 모델 이동)를 잽니다.
학습 설정은 cv.py와 같습니다(50 epoch, early stopping 없음, last.pt).

사용: python -m src.models.nested --tag p3_coco_v2b --arch yolov8s.yaml --data_sfx _v2b --outer 2,3
출력: artifacts/preds/{tag}/nested_k{k}_j{j}.csv (열 part = inner | outer)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.models.predict import read_list, yolo_predict
from src.models.yolo import train

Y = ROOT / "artifacts" / "yolo"


def build_lists(k, j, sfx, split):
    """f{k}{sfx}_train 목록에서 fold j 영상의 사본을 뺀 안쪽 학습 목록과 yaml을 만듭니다."""
    src = read_list(f"f{k}{sfx}_train")
    keep = [p for p in src if split[Path(p).stem] not in (f"f{k}", f"f{j}")]
    assert keep and len(keep) < len(src), (k, j, len(keep), len(src))
    lst = Y / "lists" / f"nest_k{k}_j{j}{sfx}_train.txt"
    lst.write_text("\n".join(keep) + "\n")
    yml = Y / f"nest_k{k}_j{j}{sfx}.yaml"
    yml.write_text(f"path: {Y.resolve()}\ntrain: {lst.resolve()}\nval: {(Y / 'lists' / f'f{j}_val.txt').resolve()}\n"
                   "names:\n  0: defect\n")
    return yml, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3_coco_v2b")
    ap.add_argument("--arch", default="yolov8s.yaml")
    ap.add_argument("--data_sfx", default="_v2b")
    ap.add_argument("--outer", default="2,3", help="1객체 영상이 있는 fold(2, 3)만 기본으로 사용")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config()
    split = json.loads((cfg["paths"]["splits"] / "split_v1.json").read_text(encoding="utf-8"))["assignment"]
    out = ROOT / "artifacts" / "preds" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in a.outer.split(",")]:
        for j in [f for f in range(4) if f != k]:
            yml, n = build_lists(k, j, a.data_sfx, split)
            name = f"{a.tag}_nest_k{k}_j{j}"
            run = train(yml, name, epochs=a.epochs, arch=a.arch, pretrained=True, seed=cfg["seed"])
            w = run / "weights" / "last.pt"
            pi, _ = yolo_predict(w, read_list(f"f{j}_val"))
            po, _ = yolo_predict(w, read_list(f"f{k}_val"))
            pd.concat([pi.assign(part="inner", fold=j), po.assign(part="outer", fold=k)]).to_csv(
                out / f"nested_k{k}_j{j}.csv", index=False)
            print(json.dumps(dict(outer=k, inner=j, n_train=n, run=str(run.relative_to(ROOT)))), flush=True)


if __name__ == "__main__":
    main()
