"""공통 평가 모듈 단위 테스트: 손 계산 사례 + pycocotools 대조. 실행: python -m tests.test_metrics"""
import contextlib
import io

import numpy as np
import pandas as pd

from src.eval import metrics as M


def test_hand():
    gts = pd.DataFrame([("a", 0, 0, 10, 10), ("a", 20, 20, 30, 30), ("b", 0, 0, 10, 10)], columns=["image_id", "x1", "y1", "x2", "y2"])
    preds = pd.DataFrame([("a", 0, 0, 10, 10, .9), ("a", 50, 50, 60, 60, .8), ("a", 21, 21, 31, 31, .7), ("b", 5, 5, 15, 15, .6)],
                         columns=["image_id", "x1", "y1", "x2", "y2", "score"])
    m, n = M.match(preds, gts, ["a", "b"], 0.5)
    assert n == 3 and m.tp.tolist() == [True, False, True, False]  # (21..31)∩(20..30): IoU 81/119=0.68; b: 25/175
    r = M.at_threshold(m, n, 2, 0.65)
    assert (r["TP"], r["FP"], r["FN"]) == (2, 1, 1) and abs(r["FPPI"] - 0.5) < 1e-9
    s = M.image_scores(preds, gts, ["a", "b"])
    assert s.set_index("image_id").loc["a", "bg_max"] == 0.8 and s.set_index("image_id").loc["b", "bg_max"] == 0.0


def coco_ap(preds, gts, ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    imgs = [{"id": k, "width": 400, "height": 400} for k in range(len(ids))]
    idmap = {i: k for k, i in enumerate(ids)}
    anns = [{"id": n + 1, "image_id": idmap[r.image_id], "category_id": 1, "bbox": [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1],
             "area": (r.x2 - r.x1) * (r.y2 - r.y1), "iscrowd": 0} for n, r in enumerate(gts.itertuples())]
    gt = COCO()
    gt.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "d"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes([{"image_id": idmap[r.image_id], "category_id": 1, "bbox": [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1],
                          "score": r.score} for r in preds.itertuples()])
        e = COCOeval(gt, dt, "bbox")
        e.params.areaRng = [[0, 1e10]] * 4
        e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[1], e.stats[0]  # AP50, AP50:95


def test_vs_coco(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"i{k}" for k in range(30)]
    g, p = [], []
    for i in ids:
        for _ in range(rng.integers(1, 4)):
            x, y = rng.uniform(20, 380, 2); w, h = rng.uniform(6, 16, 2)
            g.append((i, x, y, x + w, y + h))
            if rng.random() < 0.85:
                dx, dy = rng.normal(0, 2, 2); dw, dh = rng.normal(0, 2, 2)
                p.append((i, x + dx, y + dy, x + dx + w + dw, y + dy + h + dh, rng.uniform(.2, 1)))
        for _ in range(rng.integers(0, 3)):
            x, y = rng.uniform(20, 380, 2)
            p.append((i, x, y, x + 10, y + 10, rng.uniform(0, .7)))
    gts = pd.DataFrame(g, columns=["image_id", "x1", "y1", "x2", "y2"])
    preds = pd.DataFrame(p, columns=["image_id", "x1", "y1", "x2", "y2", "score"])
    ours = M.summarize(preds, gts, ids)
    c50, c5095 = coco_ap(preds, gts, ids)
    assert abs(ours["AP50"] - c50) < 1e-6, (ours["AP50"], c50)
    assert abs(ours["AP50_95"] - c5095) < 1e-6, (ours["AP50_95"], c5095)
    return ours["AP50"], c50, ours["AP50_95"], c5095


if __name__ == "__main__":
    test_hand()
    for s in range(3):
        print("seed", s, "ours/coco AP50, AP50:95:", [round(v, 6) for v in test_vs_coco(s)])
    print("metrics tests PASS")
