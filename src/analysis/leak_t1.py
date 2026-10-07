"""T1 흔적 판별력 시험: 마킹을 지운 자리(실제 링)와 decoy를 지운 자리를 구분할 수 있는가?

- 자리(site): 실제 링 = 원본 마킹 연결요소 중심(분석 전용), decoy = decoys_k0.json 중심
- 특징: 중심에서 체스판 거리 d=5..16 고리별 (평균 밝기 − 바깥 기준 d=17..20 평균),
        고리별 고주파 잔차의 강건 표준편차. 회전 불변이며 d≤4(이물 코어)는 제외합니다.
- 판별기: 로지스틱 회귀, HGB. 날짜 GroupKFold(5) OOF AUC
- 합격 기준(사전 고정): 기본 전처리의 실제 vs 띠 끝 decoy AUC ≤ 0.60
- 양성 대조: 원본 그레이(마킹 잔존), 팽창 0px → 높은 AUC가 나와야 시험이 민감하다는 근거가 됩니다.

출력: outputs/tables/leak_t1.json, outputs/figures/leak_t1_profiles.png
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data import marking as mk
from src.data.preprocess import image_rng

R = 20
_yy, _xx = np.mgrid[-R:R + 1, -R:R + 1]
CHEB = np.maximum(np.abs(_yy), np.abs(_xx))
DS = list(range(5, 17))


def site_features(img, resid, cx, cy):
    x, y = int(round(cx)), int(round(cy))
    H, W = img.shape
    if x - R < 0 or y - R < 0 or x + R >= W or y + R >= H:
        return None
    win = img[y - R:y + R + 1, x - R:x + R + 1].astype(np.float32)
    res = resid[y - R:y + R + 1, x - R:x + R + 1]
    ref = win[(CHEB >= 17) & (CHEB <= 20)].mean()
    f = []
    for d in DS:
        sel = CHEB == d
        v = res[sel]
        f += [win[sel].mean() - ref, 1.4826 * np.median(np.abs(v - np.median(v)))]
    return np.array(f, np.float32)


def real_sites(a):
    lab, _ = ndimage.label(mk.marking_mask(a), structure=np.ones((3, 3)))
    return [((s[1].start + s[1].stop - 1) / 2, (s[0].start + s[0].stop - 1) / 2) for s in ndimage.find_objects(lab)]


def variant_image(name, a, cfg, iid, decoy_xy, path):
    """name: default(저장된 학습 사본) | raw(가이드북식 그레이 변환, 마킹 잔존) | d0 | telea_nonoise"""
    proc = cfg["paths"]["processed"]
    if name == "default":
        return cv2.imread(str(proc / "train_k0" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
    if name == "raw":
        return np.asarray(Image.open(path).convert("L"))
    rng = image_rng(cfg["seed"], iid, 0)
    real = mk.marking_mask(a)
    if name == "d0":
        return mk.apply_decoys(mk.gray_of(a), real, decoy_xy, rng, dilate_px=0, method="telea_n")[0]
    if name == "telea_nonoise":
        return mk.apply_decoys(mk.gray_of(a), real, decoy_xy, rng, dilate_px=2, method="telea")[0]
    raise ValueError(name)


def oof_auc(X, y, groups, seed):
    out = {}
    for name, model in (("logreg", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
                        ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed))):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            model.fit(X[tr], y[tr])
            p[te] = model.predict_proba(X[te])[:, 1]
        out[name] = round(float(roc_auc_score(y, p)), 4)
    return out


def main():
    cfg = load_config()
    it, proc = cfg["paths"]["interim"], cfg["paths"]["processed"]
    idx = pd.read_csv(it / "index.csv")
    lab = idx[idx.is_canonical & idx.labeled]
    decoys = json.loads((proc / "decoys_k0.json").read_text(encoding="utf-8"))
    variants = ["default", "telea_nonoise", "d0", "raw"]
    rows = {v: [] for v in variants}
    for r in lab.itertuples():
        a = mk.read_index(cfg["data_root"] / r.file)
        ds = decoys[r.image_id]
        dxy = [(d["x"], d["y"]) for d in ds]
        sites = [(x, y, "real") for x, y in real_sites(a)] + [(d["x"], d["y"], d["kind"]) for d in ds]
        for v in variants:
            # raw에는 decoy가 없으므로 같은 decoy 자리의 원본 픽셀을 씁니다
            img = variant_image(v, a, cfg, r.image_id, dxy, cfg["data_root"] / r.file)
            resid = img.astype(np.float32) - cv2.blur(img.astype(np.float32), (7, 7))
            for x, y, kind in sites:
                f = site_features(img, resid, x, y)
                if f is not None:
                    rows[v].append((r.image_id, r.date, r.machine, kind, f))
    result = {}
    prof = {}
    for v in variants:
        kinds = np.array([k for *_, k, _ in rows[v]])
        X = np.stack([f for *_, f in rows[v]])
        groups = np.array([d for _, d, *_ in rows[v]])
        y = (kinds == "real").astype(int)
        res = {"n_real": int(y.sum()), "n_decoy": int((1 - y).sum())}
        sel = (kinds == "real") | (kinds == "band_end")
        res["real_vs_band_end"] = oof_auc(X[sel], y[sel], groups[sel], cfg["seed"])
        res["real_vs_all_decoys"] = oof_auc(X, y, groups, cfg["seed"])
        result[v] = res
        prof[v] = {k: X[kinds == k][:, 0::2].mean(0) for k in ("real", "band_end", "product")}
        print(v, json.dumps(res), flush=True)
    best = max(result["default"]["real_vs_band_end"].values())
    result["criterion"] = {"rule": "default real_vs_band_end AUC <= 0.60 (max over classifiers)", "value": best,
                           "pass": bool(best <= 0.60)}
    tab = cfg["paths"]["tables"]
    (tab / "leak_t1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.2), sharey=True)
    for ax, v in zip(axes, variants):
        for k, c in (("real", "C3"), ("band_end", "C0"), ("product", "C2")):
            ax.plot(DS, prof[v][k], "o-", c=c, ms=3, label=k)
        auc = max(result[v]["real_vs_band_end"].values())
        ax.set(title=f"{v}  (AUC real vs band-end {auc:.2f})", xlabel="chessboard distance from site (px)")
        ax.axhline(0, c="gray", lw=.6)
    axes[0].set_ylabel("mean gray − ref (d17–20)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / "leak_t1_profiles.png", dpi=150)
    print(json.dumps(result["criterion"]))


if __name__ == "__main__":
    main()
