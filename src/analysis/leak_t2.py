"""T2 누수 크기 진단 (진단 전용 — 최종 모델 성능이 아님).

(a) 표시만 쓰는 탐지기: 마킹 연결요소 중심 ± 축척×크기 박스(축척은 다른 fold에서 F1 최대로 적합), dev OOF
(b) Raw 학습 모델(fold 3, 마킹 잔존 그레이 영상으로 학습)을 raw 영상 vs clean 영상에서 평가
(c) 주모델(fold 3)을 clean vs raw 영상에서 평가 — 표시가 다시 보여도 점수가 부풀지 않는지
출력: outputs/tables/leak_t2.json
"""
import json

import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import ROOT, load_config
from src.data import marking as mk
from src.eval import metrics as M
from src.eval.compare import load_gt
from src.models.predict import yolo_predict

Y = ROOT / "artifacts" / "yolo"


def marking_boxes(cfg, canon, ids):
    rows = []
    for iid in ids:
        a = mk.read_index(cfg["data_root"] / canon.loc[iid, "file"])
        lab, _ = ndimage.label(mk.marking_mask(a), structure=np.ones((3, 3)))
        for s in ndimage.find_objects(lab):
            rows.append((iid, (s[1].start + s[1].stop) / 2, (s[0].start + s[0].stop) / 2, s[1].stop - s[1].start, s[0].stop - s[0].start))
    return pd.DataFrame(rows, columns=["image_id", "cx", "cy", "w", "h"])


def scaled(mb, k):
    return pd.DataFrame(dict(image_id=mb.image_id, x1=mb.cx - mb.w * k / 2, y1=mb.cy - mb.h * k / 2,
                             x2=mb.cx + mb.w * k / 2, y2=mb.cy + mb.h * k / 2, score=1.0))


def main():
    cfg = load_config()
    gts, meta = load_gt(cfg)
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].set_index("image_id")
    dev = meta[meta.split != "test"]
    res = {}
    mb = marking_boxes(cfg, canon, list(dev.index))
    parts, scales = [], {}
    for f in range(4):
        tr = list(dev[dev.split != f"f{f}"].index)
        te = list(dev[dev.split == f"f{f}"].index)
        best = max(np.linspace(0.3, 1.0, 15), key=lambda k: M.at_threshold(*M.match(scaled(mb[mb.image_id.isin(tr)], k), gts, tr, 0.5), len(tr), 0.5)["F1"])
        scales[f] = float(best)
        parts.append(scaled(mb[mb.image_id.isin(te)], best))
    pm = pd.concat(parts)
    ids = list(dev.index)
    s = M.summarize(pm, gts, ids, (0.5,))
    res["marking_only_dev_oof"] = dict(scales=scales, AP50=s["AP50"], AP50_95=s["AP50_95"], **{k: s["at"][0][k] for k in ("precision", "recall", "F1", "FPPI")})
    f3 = list(dev[dev.split == "f3"].index)
    for model, run in (("raw_trained", "raw_p2_coco_f3"), ("main_model", "p2_coco_f3")):
        w = ROOT / "artifacts" / "runs" / run / "weights" / "last.pt"
        if not w.exists():
            continue
        for img in ("raw", "clean"):
            pr, _ = yolo_predict(w, [Y / "images" / img / f"{i}.png" for i in f3], allow_raw=True)
            s = M.summarize(pr, gts, f3, (0.25,))
            res[f"{model}_on_{img}_fold3"] = dict(AP50=s["AP50"], AP50_95=s["AP50_95"], R_at_FPPI_0_1=M.recall_at_fppi(pr, gts, f3, 0.1)["recall"],
                                                 **{k: s["at"][0][k] for k in ("precision", "recall", "FPPI")})
    (cfg["paths"]["tables"] / "leak_t2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
