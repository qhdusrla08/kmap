"""조건별 미검(FN)·과검(FP) 분석 — dev OOF 예측 기준.

출력 (outputs/tables, figures)
  errors_{tag}_bins.csv        조건 변수 구간별 n, recall, FN, Wilson 95% CI, 평균 점수
  errors_{tag}_iou.csv         IoU 기준별 recall·AP (박스 관례 민감도)
  errors_{tag}_regression.csv  매칭 점수(logit)의 표준화 회귀계수, 날짜 군집 부트스트랩 CI
  errors_{tag}_fp.csv          운영 임계값 이상 FP의 유형·위치
  errors_{tag}_fp_audit.png    배경 FP 전체 맥락(미라벨 이물 여부 수작업 감사용) + 위치 오차 FP 상위 예시
"""
import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import structure as st
from src.eval import metrics as M
from src.eval.compare import load_gt

Y = ROOT / "artifacts" / "yolo"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bin_table(df, thr):
    specs = {
        "core_depth": [0, 40, 45, 50, 55, 100], "weber": [0, 0.45, 0.5, 0.55, 0.6, 1.0],
        "core_area": [0, 2, 3, 4, 10], "box_area": [0, 80, 100, 130, 170, 500],
        "dist_band_end": [0, 8, 11, 15, 100], "ana_mark_overlap": [-0.01, 0.0, 0.1, 0.2, 0.3, 1.0],
        "dist_edge": [0, 45, 55, 65, 200], "bg_local": [0, 80, 90, 100, 200],
    }
    cats = ["machine", "month", "n_obj", "col_rank", "slot4h", "bg_type", "width"]
    rows = []
    for v, edges in specs.items():
        b = pd.cut(df[v], edges)
        for lev, d in df.groupby(b, observed=True):
            rows.append(_row(v, str(lev), d, thr))
    for v in cats:
        for lev, d in df.groupby(v):
            rows.append(_row(v, str(lev), d, thr))
    return pd.DataFrame(rows)


def _row(var, lev, d, thr):
    k = int((d.match_score >= thr).sum())
    lo, hi = wilson(k, len(d))
    return dict(variable=var, level=lev, n=len(d), recall=k / max(len(d), 1), FN=len(d) - k, ci_lo=lo, ci_hi=hi,
                mean_score=float(d.match_score.mean()), min_score=float(d.match_score.min()))


def regression(df, groups, seed, n_boot=500):
    X = pd.DataFrame({
        "core_depth": df.core_depth, "core_area": df.core_area, "box_area": df.box_area, "bg_local": df.bg_local,
        "dist_band_end": df.dist_band_end.fillna(df.dist_band_end.median()),
        "machine3": (df.machine == 3).astype(float), "machine2": (df.machine == 2).astype(float),
        "three_obj": (df.n_obj >= 2).astype(float), "ana_mark_overlap": df.ana_mark_overlap,
    })
    X["depth_x_machine3"] = (X.core_depth - X.core_depth.mean()) * X.machine3
    Z = (X - X.mean()) / X.std().replace(0, 1)
    yv = np.log(np.clip(df.match_score, 1e-3, 1 - 1e-3) / (1 - np.clip(df.match_score, 1e-3, 1 - 1e-3)))
    A = np.c_[np.ones(len(Z)), Z.to_numpy()]
    coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        sel = np.concatenate([np.where(groups == g)[0] for g in pick])
        boots.append(np.linalg.lstsq(A[sel], yv.to_numpy()[sel], rcond=None)[0])
    boots = np.array(boots)
    return pd.DataFrame(dict(term=["intercept"] + list(Z.columns), coef=coef,
                             ci_lo=np.percentile(boots, 2.5, 0), ci_hi=np.percentile(boots, 97.5, 0)))


def fp_table(preds, gts, ids, thr):
    m, _ = M.match(preds, gts, ids, 0.5)
    p = preds.sort_values("score", ascending=False).reset_index(drop=True)
    rows = []
    G = {i: d[["x1", "y1", "x2", "y2"]].to_numpy(float) for i, d in gts[gts.image_id.isin(ids)].groupby("image_id")}
    cache = {}
    for i, d in p[p.score >= thr].groupby("image_id"):
        g = G.get(i, np.zeros((0, 4)))
        clean = cv2.imread(str(Y / "images" / "clean" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        prod = st.product_mask(clean)
        ed = st.edge_distance(prod)
        B = st.bands(clean, prod)
        band_any = np.zeros_like(prod)
        for b in B:
            band_any |= b["mask"]
        for r in d.itertuples():
            box = np.array([[r.x1, r.y1, r.x2, r.y2]])
            iou = M.iou_matrix(box, g)[0] if len(g) else np.zeros(0)
            if len(iou) and iou.max() >= 0.5:
                continue  # TP 후보 (일대일 매칭 여부는 match에서 판정, 중복 검출은 아래에서 'duplicate')
            cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
            gc = np.c_[(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2] if len(g) else np.zeros((0, 2))
            cheb = np.abs(gc - [cx, cy]).max(1).min() if len(gc) else np.inf
            xi, yi = int(np.clip(cx, 0, clean.shape[1] - 1)), int(np.clip(cy, 0, clean.shape[0] - 1))
            where = "outside_product" if not prod[yi, xi] else ("band" if band_any[yi, xi] else "product")
            dend = min((np.hypot(cx - e[0], cy - e[1]) for b in B for e in b["ends"]), default=np.nan)
            rows.append(dict(image_id=i, score=r.score, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
                             type="localization" if cheb <= 8 else "background", where=where,
                             dist_edge=float(ed[yi, xi]), dist_band_end=float(dend), max_iou=float(iou.max()) if len(iou) else 0.0))
    return pd.DataFrame(rows)


def center_offsets(preds, gts, ids, thr):
    """운영 임계값 이상이고 IoU ≥ 0.5로 정답과 겹치는 예측(정답마다 최고 점수 1개)의 중심이
    ① 정답 박스 중심 ② 코어 연결요소 중심(core_stats와 같은 정의, clean 영상)에서 얼마나 떨어졌는지(px)."""
    P = preds[preds.score >= thr]
    Pg = {i: d for i, d in P.groupby("image_id")}
    to_gt, to_core = [], []
    for i, g in gts[gts.image_id.isin(ids)].groupby("image_id"):
        d = Pg.get(i)
        if d is None:
            continue
        img = cv2.imread(str(Y / "images" / "clean" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        B = d[["x1", "y1", "x2", "y2"]].to_numpy()
        for r in g.itertuples():
            iou = M.iou_matrix([[r.x1, r.y1, r.x2, r.y2]], B)[0]
            if iou.max() < 0.5:
                continue
            b = B[int(np.argmax(np.where(iou >= 0.5, d.score.to_numpy(), -1)))]
            pc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            gc = np.array([(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2])
            to_gt.append(float(np.hypot(*(pc - gc))))
            H, W = img.shape
            x0, y0 = int(max(gc[0] - 8, 0)), int(max(gc[1] - 8, 0))
            patch = img[y0:int(min(gc[1] + 9, H)), x0:int(min(gc[0] + 9, W))].astype(np.float32)
            bg, mn = float(np.median(patch)), float(patch.min())
            from scipy import ndimage
            lab, n = ndimage.label(patch < bg - max(6.0, 0.5 * (bg - mn)))
            iy, ix = np.unravel_index(np.argmin(patch), patch.shape)
            if n and lab[iy, ix]:
                ys, xs = np.where(lab == lab[iy, ix])
                cc = np.array([x0 + xs.mean() + 0.5, y0 + ys.mean() + 0.5])  # 픽셀 중심 좌표
                to_core.append(float(np.hypot(*(pc - cc))))
    q = lambda v: dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)), p90=float(np.percentile(v, 90)))
    return dict(threshold=thr, pred_to_gt_center=q(to_gt), pred_to_core_center=q(to_core))


def fp_audit(fp, gts, out, n_bg=4, n_loc=8):
    """FP 감사 그림: ① 배경 FP(전체 영상 맥락, 미라벨 이물 여부 판정용) ② 위치 오차 FP 상위 예시(같은 이물, IoU < 0.5).
    빨강 = 예측, 초록 = TXT 정답. 빈 칸 없이 실제 개수만큼 그립니다."""
    from src.report.figures_report import STATUS
    bg = fp[fp.type == "background"].sort_values("score", ascending=False)
    lo = fp[fp.type == "localization"].sort_values("score", ascending=False)
    sections = [(k, d.head(n), len(d)) for k, d, n in (("bg", bg, n_bg), ("loc", lo, n_loc)) if len(d)]
    if not sections:
        return
    G = {i: d[["x1", "y1", "x2", "y2"]].to_numpy(float) for i, d in gts.groupby("image_id")}
    heights = [3.8 if k == "bg" else 2.5 * int(np.ceil(len(d) / 4)) + 0.3 for k, d, _ in sections]
    fig = plt.figure(figsize=(10, sum(heights) + 0.2))
    subs = np.atleast_1d(fig.subfigures(len(sections), 1, height_ratios=heights))

    def box(ax, b, color, lw):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw))

    for sf, (kind, d, total) in zip(subs, sections):
        ncol = len(d) if kind == "bg" else min(len(d), 4)
        nrow = int(np.ceil(len(d) / ncol))
        axes = np.atleast_1d(sf.subplots(nrow, ncol)).ravel()
        for ax, r in zip(axes, d.itertuples()):
            img = cv2.imread(str(Y / "images" / "clean" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray", vmin=0, vmax=243, interpolation="nearest")
            for g in G.get(r.image_id, []):
                box(ax, g, STATUS["pass"], 1.0 if kind == "bg" else 1.4)
            box(ax, (r.x1, r.y1, r.x2, r.y2), STATUS["reject"], 1.0 if kind == "bg" else 1.4)
            if kind == "bg":
                ax.set_title(f"{r.image_id}  score {r.score:.2f}  ({r.where})", fontsize=7)
            else:
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                ax.set_xlim(cx - 16, cx + 16); ax.set_ylim(cy + 16, cy - 16)
                ax.set_title(f"score {r.score:.2f}  IoU {r.max_iou:.3f}\n{r.image_id}", fontsize=7)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(d):]:
            ax.remove()
        sf.suptitle(f"Background FP: {len(d)} of {total} shown (full image; red = prediction, green = TXT label)" if kind == "bg" else
                    f"Localization FP (same object, IoU < 0.5): top {len(d)} of {total} by score", fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    dev = meta[meta.split != "test"]
    ids = list(dev.index)
    preds = pd.read_csv(ROOT / "artifacts" / "preds" / a.tag / "oof.csv")
    thr = a.thr
    if thr is None:
        t = pd.read_csv(cfg["paths"]["tables"] / "model_compare_oof.csv")
        thr = float(t[(t.model == a.tag) & (t.subset == "all")].thr_F1.iloc[0])
    cond = pd.read_csv(cfg["paths"]["interim"] / "conditions.csv")
    hits = M.object_hits(preds, gts, ids)
    df = cond.merge(hits, on=["image_id", "obj"], how="inner")
    tab, fig = cfg["paths"]["tables"], cfg["paths"]["figures"]
    bt = bin_table(df, thr)
    bt.to_csv(tab / f"errors_{a.tag}_bins.csv", index=False)
    iou_rows = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        m, n = M.match(preds, gts, ids, t)
        r = M.at_threshold(m, n, len(ids), thr)
        iou_rows.append(dict(iou=t, AP=M.average_precision(m, n), recall=r["recall"], precision=r["precision"], FPPI=r["FPPI"]))
    pd.DataFrame(iou_rows).to_csv(tab / f"errors_{a.tag}_iou.csv", index=False)
    reg = regression(df, df.image_id.map(meta.date).to_numpy(), cfg["seed"])
    reg.to_csv(tab / f"errors_{a.tag}_regression.csv", index=False)
    (tab / f"errors_{a.tag}_center.json").write_text(json.dumps(center_offsets(preds, gts, ids, thr), indent=2), encoding="utf-8")
    fp = fp_table(preds, gts, ids, thr)
    fp.to_csv(tab / f"errors_{a.tag}_fp.csv", index=False)
    fp_audit(fp, gts[gts.image_id.isin(ids)], fig / f"errors_{a.tag}_fp_audit.png")
    fn = df[df.match_score < thr]
    print(f"threshold {thr:.3f}: objects {len(df)}, FN {len(fn)}, FP {len(fp)} "
          f"(localization {int((fp.type == 'localization').sum()) if len(fp) else 0}, background {int((fp.type == 'background').sum()) if len(fp) else 0})")
    print(pd.DataFrame(iou_rows).round(3).to_string(index=False))
    print(bt[bt.FN > 0].sort_values("recall").head(15).round(3).to_string(index=False))
    print(reg.round(3).to_string(index=False))
    if len(fp):
        print(fp.groupby(["type", "where"]).size().to_string())


if __name__ == "__main__":
    main()
