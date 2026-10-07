"""전처리 이후에 그리는 데이터·전처리 그림 (보고서 제1장). 모두 이 코드로 다시 생성합니다.

  data_time_structure.png    촬영 시각 분포 + 월별 제품 유형(영상에서 검출한 띠 수)
  qc_inpaint_methods.png     표시 제거 방식 비교(원본 / telea / telea_n(채택) / 방향 보간 / 바깥 채우기)
  qc_decoy_examples.png      가짜 표시(decoy) 위치: 평가 입력(clean)과 학습 사본(k0)
  qc_transplant_examples.png 흔적-이물 독립화: 링 없는 이식(이물 O·흔적 X)과 코어 삭제 흔적 음성(흔적 O·이물 X, v2b)

그림 안 글자는 영어로 둡니다(한글 글꼴이 없는 환경에서도 깨지지 않게). 설명은 보고서 캡션에 씁니다.
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import marking as mk
from src.report.figures_report import INK, INK2, MUTED, STATUS, SURF, style

BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GT = STATUS["pass"]
KIND_COLOR = {"band_end": ORANGE, "band": AMBER, "product": BLUE, "background": "#9b59b6"}


def load(cfg):
    it = cfg["paths"]["interim"]
    idx = pd.read_csv(it / "index.csv", parse_dates=["timestamp"])
    canon = idx[idx.is_canonical].set_index("image_id")
    objs = pd.read_csv(it / "objects.csv")
    split = json.loads((cfg["paths"]["splits"] / "split_v1.json").read_text(encoding="utf-8"))["assignment"]
    S = pd.read_csv(it / "structure.csv").set_index("image_id")
    return canon, objs, split, S


def pick(canon, split, machine, n_obj, rng, month=None):
    pool = [i for i, s in split.items() if s != "test" and canon.loc[i, "machine"] == machine and canon.loc[i, "n_obj"] == n_obj
            and (month is None or canon.loc[i, "timestamp"].month == month)]
    return sorted(pool)[rng.integers(len(pool))]


def box(ax, b, color, lw=1.2, ls="-"):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw, ls=ls))


def clean_axes(axes):
    for a in np.ravel(axes):
        a.set_xticks([]); a.set_yticks([])


def time_structure(cfg, canon, S, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    fig.patch.set_facecolor(SURF)
    h = canon.timestamp.dt.hour.value_counts().reindex(range(24), fill_value=0)
    ax[0].bar(h.index, h.values, color=[BLUE if i % 4 == 0 else "#b9b8b0" for i in h.index])
    ax[0].set(xlabel="hour of day", ylabel="unique images", xticks=range(0, 24, 4))
    ax[0].set_title(f"Capture hour: {canon.slot4h.mean():.1%} at 0/4/8/12/16/20 h", fontsize=10, loc="left")
    nb = S.n_bands.reindex(canon.index).fillna(0)
    ptype = pd.Series(np.where(nb >= 2, "multi-band (3-band product)", np.where(nb == 1, "1 band", "no band detected")), index=canon.index)
    t = pd.crosstab(canon.timestamp.dt.month, ptype)[["multi-band (3-band product)", "1 band", "no band detected"]]
    bottom = np.zeros(len(t))
    for col, c in zip(t.columns, (BLUE, GREEN, MUTED)):
        ax[1].bar(t.index.astype(str), t[col], bottom=bottom, color=c, label=col, width=0.6)
        bottom += t[col].to_numpy()
    ax[1].set(xlabel="month (2020)", ylabel="unique images")
    ax[1].set_title("Product type by month (bands detected in image)", fontsize=10, loc="left")
    ax[1].legend(fontsize=8, frameon=False, loc="upper left")
    for a in ax:
        style(a)
    fig.tight_layout()
    fig.savefig(out / "data_time_structure.png", dpi=150)
    plt.close(fig)
    return t


def inpaint_methods(cfg, canon, objs, split, out, rng):
    mcfg = cfg["marking"]
    picks = [pick(canon, split, 1, 3, rng), pick(canon, split, 2, 1, rng), pick(canon, split, 3, 3, rng)]
    methods = [("raw", "raw (marking = white)"), ("telea", "Telea"), ("telea_n", "Telea + noise (adopted)"),
               ("dir", "directional"), ("outside", "outside-only fill")]
    fig, axes = plt.subplots(len(picks), len(methods), figsize=(2.3 * len(methods), 2.4 * len(picks)))
    for r, iid in enumerate(picks):
        a = mk.read_index(cfg["data_root"] / canon.loc[iid, "file"])
        g = mk.gray_of(a)
        m0 = mk.marking_mask(a)
        mask = mk.dilate(m0, mcfg["dilate_px"])
        o = objs[objs.image_id == iid]
        x1, y1 = int(max(o.x1.min() - 20, 0)), int(max(o.y1.min() - 20, 0))
        x2, y2 = int(min(o.x2.max() + 20, g.shape[1])), int(min(o.y2.max() + 20, g.shape[0]))
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sl = (slice(max(cy - side // 2, 0), cy + side // 2), slice(max(cx - side // 2, 0), cx + side // 2))
        for c, (meth, title) in enumerate(methods):
            if meth == "raw":
                img = g.copy().astype(np.uint8); img[m0] = 255
            else:
                img = mk.inpaint(g, mask, meth, mcfg["inpaint_radius"], np.random.default_rng(cfg["seed"]))
            ax = axes[r, c]
            ax.imshow(img[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK if meth != "telea_n" else ORANGE, fontweight="bold" if meth == "telea_n" else None)
            if c == 0:
                ax.set_ylabel(f"Machine {canon.loc[iid, 'machine']}", fontsize=8)
    clean_axes(axes)
    fig.suptitle("Marking removal: mask = palette index >= 244 dilated 2 px, then fill", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "qc_inpaint_methods.png", dpi=150)
    plt.close(fig)
    return picks


def decoy_examples(cfg, canon, objs, split, out, rng):
    proc = cfg["paths"]["processed"]
    dec = json.loads((proc / "decoys_k0.json").read_text(encoding="utf-8"))
    picks = [pick(canon, split, 1, 3, rng), pick(canon, split, 2, 1, rng), pick(canon, split, 3, 3, rng), pick(canon, split, 3, 1, rng)]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.0))
    for c, iid in enumerate(picks):
        for r, (sub, title) in enumerate((("clean", "evaluation input (clean)"), ("train_k0", "training copy 1 of 3: decoys drawn + erased"))):
            img = cv2.imread(str(proc / sub / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=243)
            if r == 1:
                for d in dec.get(iid, []):
                    ax.add_patch(plt.Circle((d["x"], d["y"]), 12, fill=False, ec=KIND_COLOR[d["kind"]], lw=1.0, ls="--"))
                for b in objs[objs.image_id == iid][["x1", "y1", "x2", "y2"]].to_numpy():
                    box(ax, b, GT, 1.0)
            ax.set_title(f"Machine {canon.loc[iid, 'machine']} | {iid}\n{title}", fontsize=7)
    clean_axes(axes)
    handles = [mlines.Line2D([], [], color=c, ls="none", marker="o", ms=9, mfc="none", label=f"decoy site: {k.replace('_', ' ')}")
               for k, c in KIND_COLOR.items()] + [mlines.Line2D([], [], color=GT, ls="none", marker="s", ms=7, mfc="none", label="TXT object")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "qc_decoy_examples.png", dpi=150)
    plt.close(fig)
    return picks


def yolo_boxes(path, W, H):
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4))
    a = np.loadtxt(path, ndmin=2)
    return np.c_[(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H, (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H]


def transplant_examples(cfg, canon, objs, split, out, rng):
    Y = cfg["paths"]["processed"].parent / "yolo"
    dev = sorted(i for i, s in split.items() if s != "test")
    trans, erased = [], []
    for iid in dev:
        W, H = int(canon.loc[iid, "width"]), int(canon.loc[iid, "height"])
        n = int(canon.loc[iid, "n_obj"])
        b0 = yolo_boxes(Y / "labels" / "train_k0" / f"{iid}.txt", W, H)
        if len(b0) > n:
            trans.append((iid, b0[:n], b0[n:]))
        b2 = yolo_boxes(Y / "labels" / "train_v2b_k0" / f"{iid}.txt", W, H)
        real = b0[:n]
        kept = [any(np.allclose(r, k, atol=0.6) for k in b2) for r in real]
        for r, k in zip(real, kept):
            if not k:
                erased.append((iid, r))
    sel_t = [trans[j] for j in rng.choice(len(trans), 3, replace=False)]
    sel_e = []
    for m in (1, 2, 3):  # 호기별 한 개씩
        pool = [e for e in erased if canon.loc[e[0], "machine"] == m]
        sel_e.append(pool[rng.integers(len(pool))])
    fig = plt.figure(figsize=(9.6, 13.2))
    sf = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.04)
    axa, axb = sf[0].subplots(2, 3), sf[1].subplots(2, 3)
    for c, (iid, real, tp) in enumerate(sel_t):
        img = cv2.imread(str(Y / "images" / "train_k0" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        ax = axa[0, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=243)
        for b in real:
            box(ax, b, GT, 1.0)
        for b in tp:
            box(ax, b, ORANGE, 1.2)
        ax.set_title(f"Machine {canon.loc[iid, 'machine']} | {iid}\ntraining copy: real (green) + transplanted (orange)", fontsize=7)
        b = tp[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        ax = axa[1, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=243, interpolation="nearest")
        ax.set_xlim(cx - 20, cx + 20); ax.set_ylim(cy + 20, cy - 20)
        ax.set_title("zoom: transplanted object (no marking trace)", fontsize=7)
    for c, (iid, b) in enumerate(sel_e):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for r, (sub, title) in enumerate((("train_k0", "before: real object (labeled)"), ("train_v2b_k0", "after: core erased, label removed"))):
            img = cv2.imread(str(Y / "images" / sub / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
            ax = axb[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=243, interpolation="nearest")
            ax.set_xlim(cx - 20, cx + 20); ax.set_ylim(cy + 20, cy - 20)
            box(ax, b, GT if r == 0 else MUTED, 1.2, "-" if r == 0 else ":")
            ax.set_title((f"Machine {canon.loc[iid, 'machine']} | {iid}\n" if r == 0 else "") + title, fontsize=7)
    clean_axes(axa); clean_axes(axb)
    sf[0].suptitle("(a) Object without trace: Beer-Lambert transplant to trace-free sites", fontsize=9, color=INK2, x=0.02, ha="left")
    sf[1].suptitle("(b) Trace without object (v2b): real object core erased, label removed", fontsize=9, color=INK2, x=0.02, ha="left")
    fig.savefig(out / "qc_transplant_examples.png", dpi=150)
    plt.close(fig)
    return dict(n_transplant_images=len(trans), n_erased_objects=len(erased))


def main():
    cfg = load_config()
    out = cfg["paths"]["figures"]
    canon, objs, split, S = load(cfg)
    t = time_structure(cfg, canon, S, out)
    rng = np.random.default_rng(cfg["seed"])
    p1 = inpaint_methods(cfg, canon, objs, split, out, rng)
    p2 = decoy_examples(cfg, canon, objs, split, out, rng)
    st = transplant_examples(cfg, canon, objs, split, out, rng)
    print(t.to_string())
    print(json.dumps(dict(inpaint=p1, decoy=p2, **st)))


if __name__ == "__main__":
    main()
