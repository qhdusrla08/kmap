"""OOF 기반 모델 비교표 (dev 400장, 날짜 4-fold). 테스트셋은 쓰지 않습니다.

사용: python -m src.eval.compare hgb p2_coco p2_scratch ...
출력: outputs/tables/model_compare_oof.csv, model_compare_oof.json
"""
import json
import sys

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.eval import metrics as M


def load_gt(cfg):
    objs = pd.read_csv(cfg["paths"]["interim"] / "objects.csv")
    split = json.loads((cfg["paths"]["splits"] / "split_v1.json").read_text(encoding="utf-8"))["assignment"]
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    meta = idx[idx.is_canonical & idx.labeled].set_index("image_id")[["machine", "n_obj", "date"]]
    meta["split"] = meta.index.map(split)
    return objs[["image_id", "x1", "y1", "x2", "y2"]], meta


def best_f1(m, n_gt, n_img):
    ts = np.unique(np.round(m.score.to_numpy(), 4))
    best = None
    for t in ts[::-1][:2000]:
        r = M.at_threshold(m, n_gt, n_img, t)
        if best is None or r["F1"] > best["F1"]:
            best = r
    return best


def evaluate(preds, gts, ids):
    s = M.summarize(preds, gts, ids)
    m, n = M.match(preds, gts, ids, 0.5)
    bf = best_f1(m, n, len(ids))
    imgs = M.image_scores(preds, gts, ids)
    row = dict(images=len(ids), objects=n, AP50=s["AP50"], AP75=s["AP75"], AP50_95=s["AP50_95"],
               R_at_FPPI_0_1=M.recall_at_fppi(preds, gts, ids, 0.1)["recall"],
               R_at_FPPI_0_05=M.recall_at_fppi(preds, gts, ids, 0.05)["recall"],
               F1_best=bf["F1"], P_at_F1=bf["precision"], R_at_F1=bf["recall"], thr_F1=bf["threshold"],
               FPPI_at_F1=bf["FPPI"],
               img_score_med=float(imgs.image_score.median()), bg_max_p99=float(imgs.bg_max.quantile(0.99)),
               bg_max_med=float(imgs.bg_max.median()))
    return row


def main(tags):
    cfg = load_config()
    gts, meta = load_gt(cfg)
    dev = meta[meta.split != "test"]
    rows = []
    for tag in tags:
        preds = pd.read_csv(ROOT / "artifacts" / "preds" / tag / "oof.csv")
        assert set(preds.image_id) <= set(dev.index), "OOF 예측에 dev 밖 영상이 섞였습니다"
        rows.append(dict(model=tag, subset="all", **evaluate(preds, gts, list(dev.index))))
        for name, ids in (("one_obj", dev[dev.n_obj == 1].index), ("three_obj", dev[dev.n_obj >= 2].index)):
            rows.append(dict(model=tag, subset=name, **evaluate(preds, gts, list(ids))))
        for k in (1, 2, 3):
            rows.append(dict(model=tag, subset=f"machine{k}", **evaluate(preds, gts, list(dev[dev.machine == k].index))))
        for f in range(4):
            rows.append(dict(model=tag, subset=f"fold{f}", **evaluate(preds, gts, list(dev[dev.split == f"f{f}"].index))))
    df = pd.DataFrame(rows)
    tab = cfg["paths"]["tables"]
    out = tab / "model_compare_oof.csv"
    if out.exists():  # 다른 모델의 기존 행은 유지하고, 이번에 계산한 모델의 행만 교체합니다
        old = pd.read_csv(out)
        df = pd.concat([old[~old.model.isin(tags)], df], ignore_index=True)
    df.to_csv(out, index=False)
    df = df[df.model.isin(tags)]
    cols = ["model", "subset", "images", "objects", "AP50", "AP75", "AP50_95", "R_at_FPPI_0_1", "F1_best", "thr_F1", "FPPI_at_F1", "bg_max_p99"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
