"""미라벨 NG 영상 중 AI가 '통과'로 본 영상 감사 (정답 없음 — 진단 전용).

1) 접촉 시트(육안 판정용, clean 영상만 표시)
   - 3호기 통과 영상: top-1 박스 주변 crop / 대조군: 같은 수의 3호기 1띠 재검사 영상
   - 1호기 띠 없음(OOD) 영상: 전체 썸네일
2) top-1 박스 중심과 가장 가까운 띠 끝까지 거리 — 이물은 항상 띠 끝에 있음(dev 정답 분포와 비교)
3) [분석 전용] AI–장비 위치 일치도: 원본 팔레트 ≥244(장비 표시) 연결요소의 외접 사각형 안에 AI top-1 중심이
   있는지 집계. 규칙 선정이 끝난 뒤 계산하며, 탐지·판정·임계값·학습 어디에도 쓰지 않습니다
   (docs/decisions.md 2026-10-03). 장비 미검·과검 분석이 아니라 'AI가 장비와 같은 대상을 보았는가'의 진단입니다.
출력: outputs/figures/audit_*.png, outputs/tables/audit_pass.csv, audit_pass.json
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import ROOT, load_config
from src.data import marking as mk
from src.data import structure as st

Y = ROOT / "artifacts" / "yolo"


def band_end_dist(cx, cy, ends_json):
    ends = json.loads(ends_json) if isinstance(ends_json, str) else []
    pts = [(e[0], e[1]) for e in ends] + [(e[2], e[3]) for e in ends]
    return min(np.hypot(cx - x, cy - y) for x, y in pts) if pts else np.nan


def marking_boxes(raw_path):
    """[분석 전용] 장비 표시 연결요소의 외접 사각형 목록 (x1, y1, x2, y2)."""
    m = mk.marking_mask(mk.read_index(raw_path))
    lab, n = ndimage.label(m, np.ones((3, 3)))
    return [(s[1].start, s[0].start, s[1].stop - 1, s[0].stop - 1) for s in ndimage.find_objects(lab)]


def inside(cx, cy, boxes, pad=2):
    return any(x1 - pad <= cx <= x2 + pad and y1 - pad <= cy <= y2 + pad for x1, y1, x2, y2 in boxes)


def sheet(rows, fname, title, crop=40, scale=3, ncol=11, full=False):
    n = len(rows)
    if not n:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.25, nrow * 1.45 if not full else nrow * 1.15))
    for a in np.ravel(axs):
        a.axis("off")
    for a, r in zip(np.ravel(axs), rows):
        g = cv2.imread(str(Y / "images" / "clean" / f"{r['image_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if full:
            a.imshow(g, cmap="gray", vmin=0, vmax=243)
        else:
            cx, cy = int(r["cx"]), int(r["cy"])
            h = crop // 2
            p = np.pad(g, h, mode="edge")[cy:cy + crop, cx:cx + crop]
            a.imshow(cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cmap="gray", vmin=0, vmax=243)
            bw, bh = (r["x2"] - r["x1"]) * scale, (r["y2"] - r["y1"]) * scale
            a.add_patch(plt.Rectangle(((r["x1"] - cx + h) * scale, (r["y1"] - cy + h) * scale), bw, bh, fill=False, ec="#d03b3b", lw=0.8))
        a.set_title(f"{str(r['date'])[5:]} {r['top1_conf']:.2f}", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / fname, dpi=130)
    plt.close(fig)


def main():
    cfg = load_config()
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv", parse_dates=["timestamp"])
    canon = idx[idx.is_canonical].set_index("image_id")
    S = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    T = pd.read_csv(cfg["paths"]["tables"] / "monitoring_images.csv", index_col=0, parse_dates=["timestamp"])
    pred = pd.read_csv(cfg["paths"]["predictions"] / "unlabeled_2032.csv")
    top = pred.sort_values("conf", ascending=False).groupby("image_id").head(1).set_index("image_id")
    T = T.join(top[["x1", "y1", "x2", "y2"]])
    T["cx"], T["cy"] = (T.x1 + T.x2) / 2, (T.y1 + T.y2) / 2
    T["dist_band_end"] = [band_end_dist(r.cx, r.cy, S.band_ends.get(i)) if r.top1_conf > 0.1 else np.nan for i, r in T.iterrows()]
    # 감사 대상
    m3p = T[(T.machine == 3) & (T.zone_prior == "pass")].sort_values("timestamp")
    rng = np.random.default_rng(cfg["seed"])
    pool = T[(T.machine == 3) & (T["product"] == "1band") & (T.zone_prior == "reinspect")]
    m3c = pool.loc[rng.choice(pool.index, min(len(m3p), len(pool)), replace=False)].sort_values("timestamp")
    ood = T[(T.machine == 1) & (T["product"] == "no_band")].sort_values("timestamp")
    for df, fn, tt, full in ((m3p, "audit_m3_pass.png", "Machine 3 NG images judged 'pass' (pre-registered rule): top-1 box", False),
                             (m3c, "audit_m3_reinspect_control.png", "Control: machine 3, 1-band, 'reinspect': top-1 box", False),
                             (ood.iloc[:44], "audit_m1_ood.png", "Machine 1, no band (out of training distribution): full images", True)):
        sheet([{**r.to_dict(), "image_id": i, "date": r.timestamp.date()} for i, r in df.iterrows()], fn, tt, full=full)
    # 띠 끝 거리: dev 정답(같은 정의, conditions.csv)과 비교
    cond = pd.read_csv(cfg["paths"]["interim"] / "conditions.csv")
    ref = cond[cond.machine == 3].dist_band_end
    out = dict(n_m3_pass=len(m3p), n_m3_control=len(m3c), n_m1_ood=len(ood),
               dist_band_end_median=dict(m3_pass=float(m3p.dist_band_end.median()), m3_control=float(m3c.dist_band_end.median()),
                                         dev_gt_m3=float(ref.median())),
               dist_band_end_le_25px=dict(m3_pass=float((m3p.dist_band_end <= 25).mean()), m3_control=float((m3c.dist_band_end <= 25).mean()),
                                          dev_gt_m3=float((ref <= 25).mean())))
    # [분석 전용] AI–장비 위치 일치도 (규칙 선정 후 계산, 판정·선정 미사용)
    rows = []
    for i, r in T.iterrows():
        boxes = marking_boxes(cfg["data_root"] / canon.loc[i, "file"])
        g = cv2.imread(str(Y / "images" / "clean" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        prod = st.product_mask(g)
        mc = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
        out_of_product = float(np.mean([not prod[int(np.clip(y, 0, g.shape[0] - 1)), int(np.clip(x, 0, g.shape[1] - 1))] for x, y in mc])) if mc else np.nan
        rows.append(dict(image_id=i, machine=r.machine, product=r["product"], zone_prior=r.zone_prior, zone=r.zone,
                         top1_conf=r.top1_conf, dist_band_end=r.dist_band_end, n_mark=len(boxes),
                         ai_in_mark=bool(r.top1_conf > 0.1 and inside(r.cx, r.cy, boxes)), mark_out_of_product=out_of_product))
    A = pd.DataFrame(rows).set_index("image_id")
    A.to_csv(cfg["paths"]["tables"] / "audit_pass.csv")
    agg = A.groupby(["machine", "product", "zone_prior"]).agg(n=("ai_in_mark", "size"), ai_in_mark=("ai_in_mark", "mean"),
                                                              mark_out_of_product=("mark_out_of_product", "mean")).round(3)
    out["ai_mark_agreement_analysis_only"] = json.loads(agg.reset_index().to_json(orient="records"))
    # [분석 전용] 1호기 통과(OOD 세션) 영상의 표시 위치 분포 — 같은 자리 반복 여부
    oc = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for i in A[(A.machine == 1) & (A.zone_prior == "pass")].index
                   for x1, y1, x2, y2 in marking_boxes(cfg["data_root"] / canon.loc[i, "file"])])
    if len(oc):
        out["m1_pass_mark_centers_analysis_only"] = dict(n=len(oc), x_median=float(np.median(oc[:, 0])), y_median=float(np.median(oc[:, 1])),
                                                          x_iqr=float(np.subtract(*np.percentile(oc[:, 0], [75, 25]))),
                                                          y_iqr=float(np.subtract(*np.percentile(oc[:, 1], [75, 25]))))
    # 기준선: dev 라벨 영상에서 OOF top-1 중심의 표시 안쪽 비율
    oof = pd.read_csv(ROOT / "artifacts" / "preds" / "p3_coco_v2b" / "oof.csv").sort_values("score", ascending=False).groupby("image_id").head(1)
    base = [inside((r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, marking_boxes(cfg["data_root"] / canon.loc[r.image_id, "file"])) for r in oof.itertuples()]
    out["baseline_dev_oof_top1_in_mark"] = float(np.mean(base))
    (cfg["paths"]["tables"] / "audit_pass.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(agg.to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "ai_mark_agreement_analysis_only"}, indent=1))


if __name__ == "__main__":
    main()
