"""T8 전처리 ablation (HGB 베이스라인, CPU, dev 날짜 4-fold OOF).

변형별로 학습·평가 영상을 같은 방식으로 만들고(decoy·이식 없음 → 전처리 효과만 분리),
  (1) 정상 평가: AP50, AP50:95, F1
  (2) 흔적 의존도: GT 박스 내부(이물)만 지우고 마킹 흔적은 남긴 영상에서 원위치 발화율
를 비교합니다. 마지막 행 'main'은 기본 전처리 + decoy + 이식 학습(본 파이프라인)입니다.
출력: outputs/tables/ablation_preprocess.csv
"""
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.config import ROOT, load_config
from src.data import marking as mk
from src.data.preprocess import image_rng
from src.eval import metrics as M
from src.eval.compare import best_f1, load_gt
from src.models import baseline_hgb as hb

A = ROOT / "artifacts" / "ablation"
VARIANTS = {  # 이름: (팽창 px, 방식) — None이면 가이드북식 그레이 변환(마킹 잔존)
    "raw_gray": None, "d0_telea": (0, "telea"), "d1_telea": (1, "telea"), "d2_telea": (2, "telea"),
    "d2_telea_n": (2, "telea_n"), "d3_telea_n": (3, "telea_n"), "d2_outside": (2, "outside"), "d2_dir": (2, "dir"),
}


def make(cfg, canon, ids, gt, name, spec):
    d, de = A / name / "img", A / name / "erase"
    d.mkdir(parents=True, exist_ok=True); de.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        f = cfg["data_root"] / canon.loc[iid, "file"]
        rng = image_rng(cfg["seed"], iid, 900)
        if spec is None:
            img = np.asarray(Image.open(f).convert("L"))
        else:
            img, _ = mk.remove_marking(mk.read_index(f), spec[0], spec[1], 3, rng)
        cv2.imwrite(str(d / f"{iid}.png"), img)
        m = np.zeros(img.shape, bool)
        for b in gt[iid]:  # 이물(박스 내부)만 지움 — 박스 밖 마킹 흔적은 그대로
            x1, y1, x2, y2 = int(np.floor(b[0])) + 1, int(np.floor(b[1])) + 1, int(np.ceil(b[2])) - 1, int(np.ceil(b[3])) - 1
            m[max(y1, 0):y2, max(x1, 0):x2] = True
        cv2.imwrite(str(de / f"{iid}.png"), mk.inpaint(img, m, "telea_n", 3, rng))
    return d, de


def fit_variant(paths, gt_px, seed):
    """HGB 학습용: 라벨을 영상 경로가 아니라 GT 표에서 가져오도록 build_train을 우회."""
    X, yc, Xr, yr = [], [], [], []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = gt_px[p.stem]
        xy, f = hb.candidates(img)
        if not len(xy):
            continue
        d = np.abs(xy[:, None, :] - g[None, :, :2]).max(2)
        dmin, j = d.min(1), d.argmin(1)
        pos, neg = dmin <= 3, dmin > 8
        X.append(f[pos | neg]); yc.append(pos[pos | neg])
        if pos.any():
            gg = g[j[pos]]
            Xr.append(f[pos]); yr.append(np.c_[gg[:, 0] - xy[pos, 0], gg[:, 1] - xy[pos, 1], gg[:, 2], gg[:, 3]])
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X, yc, Xr, yr = np.concatenate(X), np.concatenate(yc).astype(int), np.concatenate(Xr), np.concatenate(yr)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed).fit(X, yc)
    regs = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(Xr, yr[:, k]) for k in range(4)]
    return dict(clf=clf, regs=regs)


def site_fire(preds, gts, ids, thr):
    h = M.object_hits(preds, gts, ids)
    return float((h.center_score >= thr).mean())


def main_rows(cfg, gts, dev):
    """본 파이프라인(기본 전처리 + decoy + 이식 [+ v2 삭제 음성])의 저장된 fold 모델을 같은 d2_telea_n 평가·삭제 영상에 적용."""
    import joblib
    rows = []
    d, de = A / "d2_telea_n" / "img", A / "d2_telea_n" / "erase"
    ids = list(dev.index)
    for tag, label in (("hgb", "main_v1(d2_telea_n+decoy+transplant)"), ("hgb_v2", "main_v2(+erased-object negatives)")):
        if not (ROOT / "artifacts" / "preds" / tag / "model_f0.joblib").exists():
            continue
        P, PE = [], []
        for f in range(4):
            m = joblib.load(ROOT / "artifacts" / "preds" / tag / f"model_f{f}.joblib")
            te = list(dev[dev.split == f"f{f}"].index)
            P.append(hb.predict(m, [d / f"{i}.png" for i in te])[0])
            PE.append(hb.predict(m, [de / f"{i}.png" for i in te])[0])
        P, PE = pd.concat(P), pd.concat(PE)
        mt, n = M.match(P, gts, ids, 0.5)
        bf = best_f1(mt, n, len(ids))
        s = M.summarize(P, gts, ids)
        rows.append(dict(variant=label, AP50=s["AP50"], AP50_95=s["AP50_95"], F1_best=bf["F1"], thr=bf["threshold"],
                         recall_at_thr=bf["recall"], erased_site_fire_rate=site_fire(PE, gts, ids, bf["threshold"]),
                         erased_site_fire_rate_at_0_5=site_fire(PE, gts, ids, 0.5)))
    return rows


def main():
    cfg = load_config()
    gts, meta = load_gt(cfg)
    dev = meta[meta.split != "test"]
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].set_index("image_id")
    ids = list(dev.index)
    gt = {i: d[["x1", "y1", "x2", "y2"]].to_numpy(float) for i, d in gts.groupby("image_id")}
    gt_px = {i: np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]] for i, b in gt.items()}
    rows = []
    for name, spec in VARIANTS.items():
        d, de = make(cfg, canon, ids, gt, name, spec)
        P, PE = [], []
        for f in range(4):
            tr = [d / f"{i}.png" for i in dev[dev.split != f"f{f}"].index]
            te = list(dev[dev.split == f"f{f}"].index)
            m = fit_variant(tr, gt_px, cfg["seed"])
            P.append(hb.predict(m, [d / f"{i}.png" for i in te])[0])
            PE.append(hb.predict(m, [de / f"{i}.png" for i in te])[0])
        P, PE = pd.concat(P), pd.concat(PE)
        mt, n = M.match(P, gts, ids, 0.5)
        bf = best_f1(mt, n, len(ids))
        s = M.summarize(P, gts, ids)
        rows.append(dict(variant=name, AP50=s["AP50"], AP50_95=s["AP50_95"], F1_best=bf["F1"], thr=bf["threshold"],
                         recall_at_thr=bf["recall"], erased_site_fire_rate=site_fire(PE, gts, ids, bf["threshold"]),
                         erased_site_fire_rate_at_0_5=site_fire(PE, gts, ids, 0.5)))
        print(json.dumps(rows[-1]), flush=True)
    rows += main_rows(cfg, gts, dev)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["tables"] / "ablation_preprocess.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
