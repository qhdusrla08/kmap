"""데이터 진단 그림과 표 (보고서 1번 문항용).

  data_label_overlay.png   TXT 박스(청록)를 원본 위에 그려 좌표 형식(cx cy w h, W×H 정규화) 확인
  data_marking_halo.png    마킹 테두리 바깥·안쪽 거리별 밝기 프로파일 (후광 점검)
  (data_time_structure.png는 띠 검출이 필요해 전처리 뒤 src/report/figures_prep.py에서 그림)
  data_bbox_size.png       호기별 TXT bbox 크기 분포
  tables/halo_profile.json 후광 수치
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from src.config import load_config


def load_tables(cfg):
    it = cfg["paths"]["interim"]
    idx = pd.read_csv(it / "index.csv", parse_dates=["timestamp"])
    objs = pd.read_csv(it / "objects.csv")
    return idx[idx.is_canonical].copy(), objs


def label_overlay(cfg, canon, objs, out):
    rng = np.random.default_rng(cfg["seed"])
    lab = canon[canon.labeled]
    picks = []
    for m in (1, 2, 3):
        for n in (3, 1):
            pool = lab[(lab.machine == m) & (lab.n_obj == n)]
            if len(pool):
                picks.append(pool.iloc[rng.integers(len(pool))])
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.4))
    for j, r in enumerate(picks):
        im = Image.open(cfg["data_root"] / r.file).convert("RGB")
        o = objs[objs.image_id == r.image_id]
        cx, cy = o.x1.mean() / 2 + o.x2.mean() / 2, o.y1.mean() / 2 + o.y2.mean() / 2
        for i, ax in enumerate(axes[:, j]):
            ax.imshow(im, interpolation="nearest")
            for _, b in o.iterrows():
                ax.add_patch(mpatches.Rectangle((b.x1 - .5, b.y1 - .5), b.x2 - b.x1, b.y2 - b.y1, fill=False, ec="cyan", lw=1))
            if i == 1:  # 확대
                ax.set_xlim(cx - 30, cx + 30)
                ax.set_ylim(cy + 45, cy - 45)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].set_title(f"Machine {r.machine} | {r.width}x{r.height}\n{r.stem}", fontsize=8)
    fig.suptitle("TXT boxes (cyan) over raw BMP (device marking in color)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "data_label_overlay.png", dpi=150)
    plt.close(fig)


def halo_profiles(cfg, canon, step=3):
    """마킹 외접 사각형 기준 거리별 밝기. 바깥: 평탄 구간(거리 4–8 표준편차<1.5)만 사용.
    안쪽: 이물이 놓이기 어려운 모서리 구역(내측 2열)에서 거리 1–4 측정."""
    mmin = cfg["marking"]["palette_min_index"]
    outer = {"bright": [], "product": []}
    inner = []
    for f in canon.file.iloc[::step]:
        a = np.asarray(Image.open(cfg["data_root"] / f)).astype(float)
        H, W = a.shape
        lab, _ = ndimage.label(a >= mmin, structure=np.ones((3, 3)))
        for sl in ndimage.find_objects(lab):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if x0 < 10 or y0 < 10 or x1 > W - 10 or y1 > H - 10 or (x1 - x0) < 16:
                continue
            rows, cols = slice(y0 + 3, y1 - 3), slice(x0 + 3, x1 - 3)
            for s in (a[rows, x0 - 8:x0][:, ::-1], a[rows, x1:x1 + 8], a[y0 - 8:y0, cols][::-1].T, a[y1:y1 + 8, cols].T):
                if (s >= mmin).any():
                    continue
                p = s.mean(0)
                if p[3:].std() < 1.5:
                    outer["bright" if p[3:].mean() > 150 else "product"].append(p)
            # 안쪽: 테두리 두께 2px → 내측 거리 d는 x0+1+d
            for (ys, xs, flip) in (
                (slice(y0 + 2, y0 + 6), slice(x0 + 2, x0 + 4), False),
                (slice(y0 + 2, y0 + 6), slice(x1 - 4, x1 - 2), False),
                (slice(y1 - 6, y1 - 2), slice(x0 + 2, x0 + 4), True),
                (slice(y1 - 6, y1 - 2), slice(x1 - 4, x1 - 2), True),
            ):
                blk = a[ys, xs]
                if (blk >= mmin).any():
                    continue
                p = blk.mean(1)
                inner.append(p[::-1] if flip else p)
    res = {}
    for k, v in outer.items():
        v = np.array(v)
        res[f"outer_{k}"] = dict(n=len(v), mean_by_dist=v.mean(0).round(2).tolist() if len(v) else [],
                                 d1_minus_far=float(np.median(v[:, 0] - v[:, 3:].mean(1))) if len(v) else None,
                                 d2_minus_far=float(np.median(v[:, 1] - v[:, 3:].mean(1))) if len(v) else None)
    v = np.array(inner)
    res["inner_corner"] = dict(n=len(v), mean_by_dist=v.mean(0).round(2).tolist(),
                               d1_minus_d34=float(np.median(v[:, 0] - v[:, 2:].mean(1))),
                               d2_minus_d34=float(np.median(v[:, 1] - v[:, 2:].mean(1))))
    return res


def plot_halo(res, out):
    from src.report.figures_report import INK2, style
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, c, lab in (("outer_product", "#2a78d6", "ring on product"), ("outer_bright", "#eb6834", "ring on bright background")):
        m = res[k]["mean_by_dist"]
        if m:
            ax[0].plot(range(1, len(m) + 1), np.array(m) - np.mean(m[3:]), "o-", c=c, label=f"{lab} (n={res[k]['n']} sides)")
    ax[0].axhline(0, c="gray", lw=.8)
    ax[0].set(xlabel="distance outside the ring (px)", ylabel="gray level vs 4-8 px (mean)")
    ax[0].set_title("Outside the ring: darker halo\nup to ~3 px", fontsize=10, loc="left")
    ax[0].legend(fontsize=8, frameon=False)
    m = res["inner_corner"]["mean_by_dist"]
    ax[1].plot(range(1, len(m) + 1), np.array(m) - np.mean(m[2:]), "o-", c="#1baf7a", label=f"inner corners (n={res['inner_corner']['n']})")
    ax[1].axhline(0, c="gray", lw=.8)
    ax[1].set(xlabel="distance inside the ring (px)", ylabel="gray level vs 3-4 px (mean)", xticks=range(1, len(m) + 1))
    ax[1].set_title("Inside the ring (corner zones): brighter at 1-2 px,\nmixed with object/band gradient (not separable)", fontsize=10, loc="left")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        style(a)
    fig.tight_layout()
    fig.savefig(out / "data_marking_halo.png", dpi=150)
    plt.close(fig)


def bbox_size(objs, canon, out):
    m = objs.merge(canon[["image_id", "machine"]], on="image_id")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bins = np.arange(4, 23)
    for k in (1, 2, 3):
        s = np.sqrt(m[m.machine == k].area_px)
        ax.hist(s, bins=bins, histtype="step", lw=1.5, label=f"Machine {k} (n={len(s)}, median {np.median(s):.1f})")
    ax.set(xlabel="sqrt(bbox area) px", ylabel="objects", title="TXT object size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "data_bbox_size.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    fig_dir, tab_dir = cfg["paths"]["figures"], cfg["paths"]["tables"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    canon, objs = load_tables(cfg)
    label_overlay(cfg, canon, objs, fig_dir)
    res = halo_profiles(cfg, canon)
    (tab_dir / "halo_profile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    plot_halo(res, fig_dir)
    bbox_size(objs, canon, fig_dir)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mean_by_dist"} for k, v in res.items()}))


if __name__ == "__main__":
    main()
