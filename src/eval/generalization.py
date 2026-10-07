"""장비·시간 일반화: 호기 hold-out(LOMO)과 전향(6–7월 → 8–9월). dev 안에서만 평가합니다.
YOLO는 src.models.holdout 결과(artifacts/preds/{tag}/{split}.csv)를 읽고, HGB는 여기서 같은 분할로 학습합니다.
출력: outputs/tables/generalization.csv
"""
import argparse

import pandas as pd

from src.config import ROOT, load_config
from src.eval import metrics as M
from src.eval.compare import load_gt
from src.models import baseline_hgb as hb

SPLITS = ("lomo_m1", "lomo_m2", "lomo_m3", "forward")


def row(model, split, pr, gts, ids, thr):
    s = M.summarize(pr, gts, ids, (thr,))
    return dict(model=model, split=split, images=len(ids), objects=s["n_objects"], AP50=s["AP50"], AP75=s["AP75"],
                AP50_95=s["AP50_95"], R_at_FPPI_0_1=M.recall_at_fppi(pr, gts, ids, 0.1)["recall"],
                thr=thr, **{k: s["at"][0][k] for k in ("precision", "recall", "F1", "FPPI")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    cmp_ = pd.read_csv(cfg["paths"]["tables"] / "model_compare_oof.csv")
    thr = lambda m: float(cmp_[(cmp_.model == m) & (cmp_.subset == "all")].thr_F1.iloc[0])
    rows = []
    for sp in SPLITS:
        ids = [l.split("/")[-1][:-4] for l in hb.lst(f"{sp}_val")]
        p = ROOT / "artifacts" / "preds" / a.tag / f"{sp}.csv"
        if p.exists():
            rows.append(row(a.tag, sp, pd.read_csv(p), gts, ids, thr(a.tag)))
        m = hb.fit(hb.lst(f"{sp}_train"), cfg["seed"])
        pr, _ = hb.predict(m, hb.lst(f"{sp}_val"))
        pr.to_csv(ROOT / "artifacts" / "preds" / "hgb" / f"{sp}.csv", index=False)
        rows.append(row("hgb", sp, pr, gts, ids, thr("hgb")))
        print(sp, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["tables"] / "generalization.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
