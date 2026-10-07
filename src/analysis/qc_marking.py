"""마킹 제거 QC: 잔존 후광과 실제 링 / decoy 흔적 비교 (T1 사전 점검).

  outputs/tables/qc_marking_residual.csv  설정별 거리 1..6px 잔존 밝기 차이 (실제 링, decoy)
  outputs/figures/qc_marking_examples.png 원본 / 마스크 / 제거 결과 / decoy 예시
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from src.config import load_config
from src.data import marking as mk


def product_centers(gray, mask, rng, n, margin=14):
    """제품 내부의 평탄한 위치 중 실제 마킹에서 떨어진 곳을 decoy 중심으로 고릅니다."""
    prod = (gray > 60) & (gray < 185) & ~mask
    prod = ndimage.binary_erosion(prod, np.ones((2 * margin + 1, 2 * margin + 1)))
    far = ndimage.distance_transform_cdt(~mask, metric="chessboard") > 2 * margin
    ys, xs = np.where(prod & far)
    if len(xs) == 0:
        return []
    k = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return list(zip(xs[k], ys[k]))


def side_profiles(img, ring, comp_boxes, max_d=6):
    """링 외접 사각형의 네 변에서 바깥쪽 거리별 평균 밝기 (평탄 구간만)."""
    a = img.astype(float)
    H, W = a.shape
    out = []
    for y0, y1, x0, x1 in comp_boxes:
        if x0 < 12 or y0 < 12 or x1 > W - 12 or y1 > H - 12:
            continue
        rows, cols = slice(y0 + 3, y1 - 3), slice(x0 + 3, x1 - 3)
        for s in (a[rows, x0 - 10:x0][:, ::-1], a[rows, x1:x1 + 10], a[y0 - 10:y0, cols][::-1].T, a[y1:y1 + 10, cols].T):
            p = s.mean(0)
            if p[6:].std() < 1.5:
                out.append(p[:max_d] - p[6:].mean())
    return out


def boxes_of(mask):
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    return [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in ndimage.find_objects(lab)]


def main(step=6):
    cfg = load_config()
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    settings = list(itertools.product([0, 1, 2, 3], ["telea_n", "telea", "dir", "outside"]))
    acc = {s: {"real": [], "decoy": [], "raw_real": []} for s in settings}
    for f in canon.file.iloc[::step]:
        a = mk.read_index(cfg["data_root"] / f)
        mask = mk.marking_mask(a)
        g0 = mk.gray_of(a)
        centers = product_centers(g0, mk.dilate(mask, 3), rng, 2)
        seed = int(rng.integers(1 << 31))
        real_boxes = boxes_of(mask)
        for dpx, meth in settings:
            r2 = np.random.default_rng(seed)
            out, full, decoy = mk.apply_decoys(g0, mask, centers, r2, dilate_px=dpx, method=meth)
            acc[(dpx, meth)]["real"] += side_profiles(out, mask, real_boxes)
            acc[(dpx, meth)]["decoy"] += side_profiles(out, decoy, boxes_of(decoy))
    rows = []
    for (dpx, meth), v in acc.items():
        for kind in ("real", "decoy"):
            p = np.array(v[kind])
            if not len(p):
                continue
            med = np.median(p, 0)
            rows.append(dict(dilate_px=dpx, method=meth, ring=kind, n_sides=len(p),
                             **{f"d{i+1}": round(float(med[i]), 2) for i in range(len(med))}))
    df = pd.DataFrame(rows)
    tab = cfg["paths"]["tables"]
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / "qc_marking_residual.csv", index=False)
    print(df.to_string(index=False))
    examples(cfg, canon)


def examples(cfg, canon):
    rng = np.random.default_rng(cfg["seed"] + 1)
    picks = [canon[(canon.machine == m) & canon.labeled].iloc[i] for m, i in ((1, 5), (2, 40), (3, 7))]
    fig, axes = plt.subplots(len(picks), 5, figsize=(15, 3.2 * len(picks)))
    for row, r in zip(axes, picks):
        a = mk.read_index(cfg["data_root"] / r.file)
        mask = mk.marking_mask(a)
        g0 = mk.gray_of(a)
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        win = (slice(max(cy - 45, 0), cy + 45), slice(max(cx - 45, 0), cx + 45))
        centers = product_centers(g0, mk.dilate(mask, 3), rng, 2)
        telea, full, _ = mk.apply_decoys(g0, mask, [], rng, 2, "telea_n")
        outside, _, _ = mk.apply_decoys(g0, mask, [], rng, 2, "outside")
        dec, full_d, decoy = mk.apply_decoys(g0, mask, centers, rng, 2, "telea_n")
        rgb = np.asarray(Image.open(cfg["data_root"] / r.file).convert("RGB"))
        panels = [(rgb[win], "raw (palette→RGB)"), (np.where(full, 255, g0)[win], "mask (2px dilated)"),
                  (telea[win], "telea_n d2 (default)"), (outside[win], "outside-fill d2 (ablation)"), (dec, "with decoys (full image)")]
        for ax, (im, title) in zip(row, panels):
            ax.imshow(im, cmap="gray", vmin=0, vmax=243, interpolation="nearest")
            ax.set_title(f"M{r.machine} {title}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        for cx2, cy2 in centers:
            row[4].add_patch(plt.Rectangle((cx2 - 12, cy2 - 12), 24, 24, fill=False, ec="yellow", lw=.6, ls=":"))
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / "qc_marking_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
