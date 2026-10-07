"""객체별 조건 변수 (docs/data_spec.md §8). 출력: artifacts/interim/conditions.csv

- 추론 시점에도 계산 가능한 변수: clean(마킹 제거) 영상과 메타데이터만 사용
- 'ana_' 접두 변수: 원본 마킹을 참조하는 사후 분석 전용 — 판정 규칙에 쓰지 않습니다
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import ROOT, load_config
from src.data import marking as mk
from src.data import structure as st

Y = ROOT / "artifacts" / "yolo"


def core_stats(img, cx, cy, mask=None):
    """GT 중심 17×17에서 코어(국소 배경 − max(6, 0.5·깊이)보다 어두운 최소점 연결요소) 통계."""
    H, W = img.shape
    x0, x1 = int(max(cx - 8, 0)), int(min(cx + 9, W))
    y0, y1 = int(max(cy - 8, 0)), int(min(cy + 9, H))
    p = img[y0:y1, x0:x1].astype(np.float32)
    valid = np.ones_like(p, bool) if mask is None else ~mask[y0:y1, x0:x1]
    v = p[valid]
    if v.size < 20:
        return dict(core_area=np.nan, core_side=np.nan, core_depth=np.nan, bg_local=np.nan, weber=np.nan, integrated_dark=np.nan, core_min=np.nan)
    bg, mn = float(np.median(v)), float(v.min())
    thr = bg - max(6.0, 0.5 * (bg - mn))
    lab, n = ndimage.label((p < thr) & valid)
    pp = np.where(valid, p, 1e9)
    iy, ix = np.unravel_index(np.argmin(pp), p.shape)
    c = lab == lab[iy, ix] if n and lab[iy, ix] else np.zeros_like(p, bool)
    ys, xs = np.where(c)
    return dict(core_area=int(c.sum()), core_side=int(max(np.ptp(ys), np.ptp(xs)) + 1) if len(xs) else 0,
                core_depth=bg - mn, bg_local=bg, weber=(bg - mn) / max(bg, 1), core_min=mn,
                integrated_dark=float((bg - p[c]).sum()))


def main():
    cfg = load_config()
    it = cfg["paths"]["interim"]
    idx = pd.read_csv(it / "index.csv", parse_dates=["timestamp"])
    canon = idx[idx.is_canonical].set_index("image_id")
    objs = pd.read_csv(it / "objects.csv")
    rows = []
    for iid, g in objs.groupby("image_id", sort=False):
        r = canon.loc[iid]
        clean = cv2.imread(str(Y / "images" / "clean" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        a = mk.read_index(cfg["data_root"] / r.file)
        real = mk.marking_mask(a)
        rmask = mk.dilate(real, cfg["marking"]["dilate_px"])
        prod = st.product_mask(clean)
        ed = st.edge_distance(prod)
        B = st.bands(clean, prod)
        band_any = np.zeros_like(prod)
        for b in B:
            band_any |= b["mask"]
        band_edge = ndimage.distance_transform_edt(~band_any) if band_any.any() else np.full(prod.shape, np.inf)
        band_in = ndimage.distance_transform_edt(band_any) if band_any.any() else np.zeros(prod.shape)
        ys_rank = g.assign(cy=(g.y1 + g.y2) / 2).cy.rank(method="first").astype(int).values
        for (k, o), rank in zip(g.iterrows(), ys_rank):
            cx, cy = (o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2
            xi, yi = int(np.clip(round(cx), 0, clean.shape[1] - 1)), int(np.clip(round(cy), 0, clean.shape[0] - 1))
            cs = core_stats(clean, cx, cy)
            raw_cs = core_stats(a.astype(np.uint8), cx, cy, mask=rmask)  # 원본 픽셀(마스크 밖)만
            dend = min((np.hypot(cx - e[0], cy - e[1]) for b in B for e in b["ends"]), default=np.nan)
            x1, y1, x2, y2 = int(np.floor(o.x1)), int(np.floor(o.y1)), int(np.ceil(o.x2)), int(np.ceil(o.y2))
            box_m = rmask[max(y1, 0):y2, max(x1, 0):x2]
            if band_any[yi, xi]:
                bg_type = "band_edge" if band_in[yi, xi] <= 2 else "band"
            else:
                bg_type = "band_edge" if band_edge[yi, xi] <= 2 else "product"
            rows.append(dict(
                image_id=iid, obj=int(o.obj), machine=int(r.machine), width=int(r.width), month=int(r.timestamp.month),
                hour=int(r.timestamp.hour), slot4h=bool(r.slot4h), n_obj=int(r.n_obj), col_rank=int(rank) if r.n_obj >= 2 else 0,
                box_w=o.w_px, box_h=o.h_px, box_area=o.area_px, box_aspect=o.w_px / max(o.h_px, 1e-6),
                **cs, dist_edge=float(ed[yi, xi]), dist_band_end=float(dend), bg_type=bg_type,
                product_bg=float(np.median(clean[prod])) if prod.any() else np.nan,
                ana_mark_overlap=float(box_m.mean()) if box_m.size else np.nan,
                ana_raw_core_depth=raw_cs["core_depth"], ana_inpaint_depth_delta=cs["core_depth"] - raw_cs["core_depth"],
            ))
    df = pd.DataFrame(rows)
    df.to_csv(it / "conditions.csv", index=False)
    print(df.describe(include="all").T[["count", "mean", "50%", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
