"""통제 스트레스 시험: 대비 감쇠 × 위치 → 탐지 한계 곡선 (dev fold 모델 → held-out fold).

실제 라벨 객체는 모두 고대비(코어 깊이 32–71)이고 제품 경계에서 37px 이상 떨어져 있어,
'작은·저대비·가장자리 이물' 조건은 데이터에 없습니다. Beer–Lambert 투과율 T를
T_α = 1 − α(1 − T)로 줄여(α=1 원본, α→0 소멸) 대비를 통제합니다.

  in_place: 원위치 객체의 대비만 감쇠 (링 흔적 있음)
  band_end / interior / edge: 링 흔적 없는 위치에 감쇠된 이물 이식 (edge = 제품 경계에서 5–12px)

출력: outputs/tables/stress_{tag}.csv (객체별), stress_{tag}_summary.csv, figures/stress_{tag}.png
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
from src.data import transplant as tp
from src.data.preprocess import INWARD_RANGE, _far, image_rng
from src.eval import metrics as M
from src.models.predict import predict_fold

Y = ROOT / "artifacts" / "yolo"
ALPHAS = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)


def attenuate(T, alpha):
    return 1 - alpha * (1 - T)


def pick_site(kind, B, prod, ed, centers, taken, machine, rng):
    lo, hi = INWARD_RANGE[machine]
    for _ in range(40):
        if kind == "band_end":
            if not B:
                return None
            b = B[rng.integers(len(B))]
            e = b["ends"][rng.integers(2)]
            ax = np.array(b["axis"])
            inward = np.sign(np.dot(np.array(b["center"]) - np.array(e), ax)) * ax
            p = tuple(np.array(e) + inward * rng.uniform(lo, hi))
        else:
            sel = (ed > 15) if kind == "interior" else ((ed >= 5) & (ed <= 12))
            ys, xs = np.where(sel)
            if not len(xs):
                return None
            k = rng.integers(len(xs))
            p = (float(xs[k]), float(ys[k]))
        if _far(p, centers, 18) and _far(p, taken, 18):
            return p
    return None


def build(cfg, ids, canon, gt, fold):
    recs = []
    dirs = {a: Y / "images" / f"eval_stress_a{int(a * 100):03d}" for a in ALPHAS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        clean = cv2.imread(str(Y / "images" / "clean" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        g = gt[iid]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in g]
        prod = st.product_mask(clean)
        ed = st.edge_distance(prod)
        B = st.bands(clean, prod)
        machine = int(canon.loc[iid, "machine"])
        rng = image_rng(cfg["seed"], iid, 700)
        ex = [tp.extract_ratio(clean, *c) for c in centers]
        # 위치는 α와 무관하게 고정 (같은 사이트에서 대비만 변화)
        sites, taken = [], []
        for kind in ("band_end", "interior", "edge"):
            p = pick_site(kind, B, prod, ed, centers, taken, machine, rng)
            j = int(rng.integers(len(g)))
            if p is not None and ex[j] is not None:
                sites.append((kind, p, j)); taken.append(p)
        for a in ALPHAS:
            img = clean.astype(np.float32)
            for j, e in enumerate(ex):  # 원위치 감쇠: I·T_α/T
                if e is None:
                    continue
                T, (kx, ky) = e
                y0, x0 = ky - tp.R, kx - tp.R
                img[y0:y0 + 2 * tp.R + 1, x0:x0 + 2 * tp.R + 1] *= attenuate(T, a) / T
                b = g[j]
                recs.append(dict(fold=fold, image_id=iid, alpha=a, kind="in_place", x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
            img = np.clip(np.round(img), 0, 243).astype(np.uint8)
            for kind, p, j in sites:
                T, (kx, ky) = ex[j]
                if not tp.paste(img, attenuate(T, a), *p):
                    continue
                b = g[j]
                bx, by = p[0] + centers[j][0] - kx, p[1] + centers[j][1] - ky
                w, h = b[2] - b[0], b[3] - b[1]
                recs.append(dict(fold=fold, image_id=iid, alpha=a, kind=kind, x1=bx - w / 2, y1=by - h / 2, x2=bx + w / 2, y2=by + h / 2))
            cv2.imwrite(str(dirs[a] / f"{iid}.png"), img)
    return pd.DataFrame(recs), dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config()
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].set_index("image_id")
    objs = pd.read_csv(cfg["paths"]["interim"] / "objects.csv")
    cond = pd.read_csv(cfg["paths"]["interim"] / "conditions.csv")
    gt = {i: d[["x1", "y1", "x2", "y2"]].to_numpy(float) for i, d in objs.groupby("image_id")}
    split = json.loads((cfg["paths"]["splits"] / "split_v1.json").read_text(encoding="utf-8"))["assignment"]
    thr = a.thr
    if thr is None:
        t = pd.read_csv(cfg["paths"]["tables"] / "model_compare_oof.csv")
        thr = float(t[(t.model == a.tag) & (t.subset == "all")].thr_F1.iloc[0])
    allrec = []
    for f in range(4):
        ids = sorted(i for i, s in split.items() if s == f"f{f}")
        recs, dirs = build(cfg, ids, canon, gt, f)
        for al, d in dirs.items():
            pr, _ = predict_fold(a.tag, f, [d / f"{i}.png" for i in ids])
            sub = recs[recs.alpha == al].reset_index(drop=True)
            for kind, s in sub.groupby("kind"):
                h = M.object_hits(pr, s[["image_id", "x1", "y1", "x2", "y2"]], ids)
                s = s.reset_index(drop=True).assign(match_score=h.match_score.values, center_score=h.center_score.values)
                allrec.append(s)
        print(f"fold {f} done", flush=True)
    df = pd.concat(allrec, ignore_index=True)
    # 원위치 객체의 원래 코어 깊이를 붙여 '유효 깊이' = α × 깊이
    cmap = cond.set_index(["image_id", "obj"]).core_depth
    df["hit"] = df.match_score >= thr
    tab = cfg["paths"]["tables"]
    df.to_csv(tab / f"stress_{a.tag}.csv", index=False)
    summ = df.groupby(["kind", "alpha"]).agg(n=("hit", "size"), recall=("hit", "mean")).reset_index()
    summ.to_csv(tab / f"stress_{a.tag}_summary.csv", index=False)
    piv = summ.pivot(index="alpha", columns="kind", values="recall").sort_index(ascending=False)
    print(piv.round(3).to_string())
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for kind in ("in_place", "band_end", "interior", "edge"):
        if kind in piv:
            ax.plot(piv.index, piv[kind], "o-", label=kind)
    ax.set(xlabel="contrast factor α (1 = original object)", ylabel=f"recall @ thr {thr:.2f}",
           title=f"Detection limit by contrast and position ({a.tag})")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / f"stress_{a.tag}.png", dpi=150)


if __name__ == "__main__":
    main()
