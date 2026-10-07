"""공통 평가 모듈 (모든 모델 동일 기준).

입력
  preds: DataFrame[image_id, x1, y1, x2, y2, score]  (원본 픽셀 좌표)
  gts:   DataFrame[image_id, x1, y1, x2, y2]          (TXT 정답)
  image_ids: 평가 대상 영상 목록 (예측이 없는 영상도 포함)

지표
  - AP@IoU: COCO 방식(점수순 일대일 매칭, 101점 보간). AP50, AP75, AP50:95
  - 임계값 t에서 TP/FP/FN, Precision, Recall, F1, FPPI(이미지당 FP)
  - 객체별: 최고 점수 매칭 여부(hit), 매칭 점수, 최대 IoU → 조건 분석용
  - 이미지별: image_score(최대 점수), bg_max(GT 주변이 아닌 예측의 최대 점수; 음성 대용)
"""
import numpy as np
import pandas as pd

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def _group(df, ids, cols):
    g = {i: np.zeros((0, len(cols))) for i in ids}
    for i, d in df[df.image_id.isin(ids)].groupby("image_id"):
        g[i] = d[cols].to_numpy(float)
    return g


def match(preds, gts, image_ids, iou_thr=0.5):
    """점수순 일대일 매칭. 반환: 예측별 (image_id, score, tp, gt_index), GT 수."""
    P = _group(preds.sort_values("score", ascending=False), image_ids, ["x1", "y1", "x2", "y2", "score"])
    G = _group(gts, image_ids, ["x1", "y1", "x2", "y2"])
    recs = []
    for i in image_ids:
        p, g = P[i], G[i]
        used = np.zeros(len(g), bool)
        ious = iou_matrix(p[:, :4], g) if len(p) and len(g) else np.zeros((len(p), len(g)))
        for k in range(len(p)):
            j = -1
            if len(g):
                cand = np.where(~used & (ious[k] >= iou_thr))[0]
                if len(cand):
                    j = int(cand[np.argmax(ious[k, cand])])
                    used[j] = True
            recs.append((i, p[k, 4], j >= 0, j))
    m = pd.DataFrame(recs, columns=["image_id", "score", "tp", "gt_index"])
    return m, sum(len(g) for g in G.values())


def average_precision(m, n_gt):
    """COCO 101점 보간 AP."""
    if n_gt == 0:
        return float("nan")
    m = m.sort_values("score", ascending=False, kind="mergesort")
    tp = m.tp.to_numpy().astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    rs = np.linspace(0, 1, 101)
    q = np.zeros(101)
    if len(rec):
        idx = np.searchsorted(rec, rs, side="left")
        ok = idx < len(prec)
        q[ok] = prec[idx[ok]]
    return float(q.mean())


def at_threshold(m, n_gt, n_images, t):
    s = m[m.score >= t]
    tp, fp = int(s.tp.sum()), int((~s.tp).sum())
    fn = n_gt - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    return dict(threshold=float(t), TP=tp, FP=fp, FN=fn, precision=p, recall=r,
                F1=2 * p * r / max(p + r, 1e-9), FPPI=fp / max(n_images, 1))


def object_hits(preds, gts, image_ids, iou_thr=0.5):
    """GT 객체별: 매칭된 예측 점수(없으면 0), 최대 IoU, 중심 포함 예측의 최대 점수."""
    m, _ = match(preds, gts, image_ids, iou_thr)
    tp = m[m.tp]
    best = {(r.image_id, int(r.gt_index)): r.score for r in tp.itertuples()}
    P = _group(preds, image_ids, ["x1", "y1", "x2", "y2", "score"])
    rows = []
    for i, d in gts[gts.image_id.isin(image_ids)].groupby("image_id"):
        p = P[i]
        for j, g in enumerate(d[["x1", "y1", "x2", "y2"]].to_numpy(float)):
            iou = iou_matrix(p[:, :4], g[None])[:, 0] if len(p) else np.zeros(0)
            cx, cy = (p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2
            inside = (cx >= g[0]) & (cx <= g[2]) & (cy >= g[1]) & (cy <= g[3]) if len(p) else np.zeros(0, bool)
            rows.append(dict(image_id=i, obj=j, match_score=best.get((i, j), 0.0),
                             max_iou=float(iou.max()) if len(iou) else 0.0,
                             center_score=float(p[inside, 4].max()) if inside.any() else 0.0))
    return pd.DataFrame(rows)


def image_scores(preds, gts, image_ids, near_px=12):
    """image_score: 영상 내 최대 점수. bg_max: 모든 GT 중심에서 near_px(체스판 거리)보다 멀고
    GT와 IoU<0.1인 예측의 최대 점수 — 같은 제품 배경을 가진 '음성 영상'의 점수 대용."""
    P = _group(preds, image_ids, ["x1", "y1", "x2", "y2", "score"])
    G = _group(gts, image_ids, ["x1", "y1", "x2", "y2"])
    rows = []
    for i in image_ids:
        p, g = P[i], G[i]
        if not len(p):
            rows.append(dict(image_id=i, image_score=0.0, bg_max=0.0, n_gt=len(g)))
            continue
        far = np.ones(len(p), bool)
        if len(g):
            pc = np.c_[(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2]
            gc = np.c_[(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2]
            cheb = np.abs(pc[:, None, :] - gc[None, :, :]).max(2)
            far = (cheb > near_px).all(1) & (iou_matrix(p[:, :4], g).max(1) < 0.1)
        rows.append(dict(image_id=i, image_score=float(p[:, 4].max()),
                         bg_max=float(p[far, 4].max()) if far.any() else 0.0, n_gt=len(g)))
    return pd.DataFrame(rows)


def summarize(preds, gts, image_ids, thresholds=(0.25,)):
    out = {}
    for t in IOU_THRS:
        m, n = match(preds, gts, image_ids, t)
        out[f"AP{int(round(t * 100))}"] = average_precision(m, n)
    out["AP50_95"] = float(np.mean([out[f"AP{int(round(t * 100))}"] for t in IOU_THRS]))
    m, n = match(preds, gts, image_ids, 0.5)
    out["n_images"], out["n_objects"] = len(image_ids), n
    out["at"] = [at_threshold(m, n, len(image_ids), t) for t in thresholds]
    return out


def recall_at_fppi(preds, gts, image_ids, fppi=0.1, iou_thr=0.5):
    """FPPI가 목표 이하가 되는 가장 낮은 점수 임계값에서의 recall."""
    m, n = match(preds, gts, image_ids, iou_thr)
    m = m.sort_values("score", ascending=False, kind="mergesort")
    fp = np.cumsum(~m.tp.to_numpy()) / max(len(image_ids), 1)
    tp = np.cumsum(m.tp.to_numpy())
    ok = np.where(fp <= fppi)[0]
    if not len(ok):
        return dict(recall=0.0, threshold=float("inf"))
    k = ok[-1]
    return dict(recall=float(tp[k] / max(n, 1)), threshold=float(m.score.iloc[k]))
