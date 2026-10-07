"""TTA(반전 4종) 기반 불확실성과 재검사 규칙 검증 — dev fold 모델 → held-out fold.

보기: 원본, 좌우, 상하, 상하좌우 반전. 각 보기의 예측을 원좌표로 되돌린 뒤
  - 양성: GT 객체별 매칭 점수(보정 전)의 보기 간 평균·표준편차
  - 음성 대용: 영상별 bg_max의 보기 간 평균·표준편차
규칙 후보: '통과 구간(p < t_low)이지만 std ≥ u'이면 재검사 → 추가 재검사율과 회수되는 미검 수
출력: outputs/tables/uncertainty_{tag}.json, uncertainty_{tag}_objects.csv
"""
import argparse
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from src.analysis.thresholds import apply_cal
from src.config import ROOT, load_config
from src.eval import metrics as M
from src.eval.compare import load_gt
from src.models.predict import predict_fold

Y = ROOT / "artifacts" / "yolo"
VIEWS = {"id": None, "h": 1, "v": 0, "hv": -1}


def unflip(pr, sizes, code):
    if code is None or not len(pr):
        return pr
    pr = pr.copy()
    W = pr.image_id.map(lambda i: sizes[i][0])
    H = pr.image_id.map(lambda i: sizes[i][1])
    if code in (1, -1):
        pr["x1"], pr["x2"] = W - pr.x2, W - pr.x1
    if code in (0, -1):
        pr["y1"], pr["y2"] = H - pr.y2, H - pr.y1
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    dev = meta[meta.split != "test"]
    thr = json.loads((cfg["paths"]["tables"] / f"thresholds_{a.tag}.json").read_text())
    cal = joblib.load(ROOT / "artifacts" / "preds" / a.tag / "calibrator.joblib")
    obj_scores, bg_scores = {}, {}
    for f in range(4):
        ids = sorted(dev[dev.split == f"f{f}"].index)
        sizes = {}
        for v, code in VIEWS.items():
            d = Y / "images" / f"eval_tta_{v}"
            d.mkdir(parents=True, exist_ok=True)
            for i in ids:
                im = cv2.imread(str(Y / "images" / "clean" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
                sizes[i] = (im.shape[1], im.shape[0])
                cv2.imwrite(str(d / f"{i}.png"), im if code is None else cv2.flip(im, code))
            pr, _ = predict_fold(a.tag, f, [d / f"{i}.png" for i in ids])
            pr = unflip(pr, sizes, code)
            pr["score"] = apply_cal(cal["model"], cal["kind"], pr.score.to_numpy()) if len(pr) else []
            h = M.object_hits(pr, gts, ids)
            for r in h.itertuples():
                obj_scores.setdefault((r.image_id, r.obj), {})[v] = r.match_score
            for r in M.image_scores(pr, gts, ids).itertuples():
                bg_scores.setdefault(r.image_id, {})[v] = r.bg_max
        print(f"fold {f} done", flush=True)
    O = pd.DataFrame([dict(image_id=k[0], obj=k[1], **v) for k, v in obj_scores.items()])
    B = pd.DataFrame([dict(image_id=k, **v) for k, v in bg_scores.items()])
    for D in (O, B):
        D["mean"] = D[list(VIEWS)].mean(1)
        D["std"] = D[list(VIEWS)].std(1)
    t_low = thr["t_low"]
    res = dict(model=a.tag, t_low=t_low, n_objects=len(O), n_images=len(B))
    res["corr_std_vs_mean_objects"] = float(np.corrcoef(O["mean"], O["std"])[0, 1])
    rules = []
    for u in (0.02, 0.05, 0.1, 0.15, 0.2):
        miss = O["id"] < t_low
        rec = miss & (O["std"] >= u)
        neg_extra = (B["id"] < t_low) & (B["std"] >= u)
        rules.append(dict(u=u, misses_at_t_low=int(miss.sum()), misses_recovered=int(rec.sum()),
                          extra_reinspect_rate_neg_proxy=float(neg_extra.mean())))
    res["rules"] = rules
    # 대안: 보기 평균 점수로 판정하면?
    res["recall_objects_id_at_t_low"] = float((O["id"] >= t_low).mean())
    res["recall_objects_tta_mean_at_t_low"] = float((O["mean"] >= t_low).mean())
    res["negproxy_fpr_id_at_t_low"] = float((B["id"] >= t_low).mean())
    res["negproxy_fpr_tta_mean_at_t_low"] = float((B["mean"] >= t_low).mean())
    tab = cfg["paths"]["tables"]
    O.to_csv(tab / f"uncertainty_{a.tag}_objects.csv", index=False)
    (tab / f"uncertainty_{a.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
