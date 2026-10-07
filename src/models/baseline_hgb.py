"""베이스라인: 고전 후보 생성 + HGB 분류 + 가변 박스 회귀 (Codex 사전 실험 이식·개선).

- 후보: 11×11 black-hat, 5×5 국소 최대, bh≥3, 영상당 최대 800개 (정답·마킹 미참조)
- 특징: 17×17 패치를 9×9로 축소한 값(81) + 평균·표준편차·bh·중심 3×3 평균 = 85 (Codex와 동일)
- 개선: 고정 10×10 박스 대신 HGB 회귀로 (dx, dy, w, h) 예측. 해상도(W, H)를 보조 입력으로 씀
- 학습: YOLO와 같은 학습 사본(decoy + 링 없는 이식) / 평가: clean 영상
- 라벨: 후보와 GT 중심의 체스판 거리 ≤3 → 양성, >8 → 음성, 그 사이는 제외

출력: artifacts/preds/hgb/oof.csv (fold 열 포함), 필요 시 dev 전체 모델로 test·unlabeled 예측
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.config import ROOT, load_config

Y = ROOT / "artifacts" / "yolo"


def candidates(gray, max_n=800):
    g = gray.astype(np.uint8)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((11, 11), np.uint8)).astype(np.float32)
    peak = (bh == maximum_filter(bh, size=5)) & (bh >= 3)
    peak[:10] = 0; peak[-10:] = 0; peak[:, :10] = 0; peak[:, -10:] = 0
    yy, xx = np.where(peak)
    order = np.argsort(-bh[yy, xx], kind="stable")[:max_n]
    xx, yy = xx[order], yy[order]
    gf = g.astype(np.float32)
    H, W = g.shape
    feats = np.empty((len(xx), 87), np.float32)
    for k, (x, y) in enumerate(zip(xx, yy)):
        p = gf[y - 8:y + 9, x - 8:x + 9]
        q = cv2.resize(p, (9, 9), interpolation=cv2.INTER_AREA)
        feats[k, :81] = (q - q.mean()).ravel() / 32
        feats[k, 81:85] = (q.mean() / 255, q.std() / 32, bh[y, x] / 32, p[7:10, 7:10].mean() / 255)
        feats[k, 85:] = (W / 400, H / 400)
    return np.c_[xx, yy].astype(np.float32), feats


def read_labels(path, W, H):
    if not Path(path).exists():
        return np.zeros((0, 4))
    a = np.loadtxt(path, ndmin=2)
    if not len(a):
        return np.zeros((0, 4))
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return np.c_[cx, cy, w, h]


def build_train(paths):
    X, yc, Xr, yr = [], [], [], []
    for p in paths:
        p = Path(p)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        g = read_labels(Y / "labels" / p.parent.name / f"{p.stem}.txt", W, H)
        xy, f = candidates(img)
        if not len(xy):
            continue
        if len(g):
            d = np.abs(xy[:, None, :] - g[None, :, :2]).max(2)
            dmin, j = d.min(1), d.argmin(1)
        else:
            dmin, j = np.full(len(xy), np.inf), np.zeros(len(xy), int)
        pos, neg = dmin <= 3, dmin > 8
        X.append(f[pos | neg]); yc.append(pos[pos | neg])
        if pos.any():
            Xr.append(f[pos])
            gg = g[j[pos]]
            yr.append(np.c_[gg[:, 0] - xy[pos, 0], gg[:, 1] - xy[pos, 1], gg[:, 2], gg[:, 3]])
    return np.concatenate(X), np.concatenate(yc).astype(int), np.concatenate(Xr), np.concatenate(yr)


def fit(paths, seed):
    X, y, Xr, yr = build_train(paths)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=seed).fit(X, y)
    regs = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(Xr, yr[:, k])
            for k in range(4)]
    return dict(clf=clf, regs=regs, n_train=int(len(y)), n_pos=int(y.sum()))


def nms(boxes, scores, thr=0.3, keep_max=20):
    from src.eval.metrics import iou_matrix
    order = np.argsort(-scores)
    keep = []
    for i in order:
        if scores[i] < 1e-3:
            break
        if keep and iou_matrix(boxes[i:i + 1], boxes[keep]).max() > thr:
            continue
        keep.append(i)
        if len(keep) >= keep_max:
            break
    return keep


def predict(model, paths):
    rows, times = [], []
    for p in paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        xy, f = candidates(img)
        if not len(xy):
            continue
        s = model["clf"].predict_proba(f)[:, 1]
        dx, dy, w, h = (r.predict(f) for r in model["regs"])
        cx, cy = xy[:, 0] + dx, xy[:, 1] + dy
        w, h = np.clip(w, 4, 30), np.clip(h, 4, 30)
        boxes = np.c_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        for i in nms(boxes, s):
            rows.append((Path(p).stem, *boxes[i], float(s[i])))
        times.append((time.perf_counter() - t0) * 1000)
    df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"])
    return df, dict(ms_per_image_mean=float(np.mean(times)))


def lst(name):
    return [l for l in (Y / "lists" / f"{name}.txt").read_text().splitlines() if l]


def main(mode="oof", variant=""):
    """variant: '' (v1 학습 사본) 또는 'v2' (실제 흔적-무이물 음성 추가)"""
    cfg = load_config()
    sfx = f"_{variant}" if variant else ""
    out = ROOT / "artifacts" / "preds" / f"hgb{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    if mode == "oof":
        parts, info = [], {}
        for f in range(4):
            m = fit(lst(f"f{f}{sfx}_train"), cfg["seed"])
            joblib.dump(m, out / f"model_f{f}.joblib")
            pr, sp = predict(m, lst(f"f{f}_val"))
            parts.append(pr.assign(fold=f))
            info[f"f{f}"] = dict(n_train=m["n_train"], n_pos=m["n_pos"], **sp)
            print(f"fold {f}", json.dumps(info[f"f{f}"]), flush=True)
        pd.concat(parts).to_csv(out / "oof.csv", index=False)
        (out / "oof_info.json").write_text(json.dumps(info, indent=2))
    elif mode == "final":
        m = fit(lst(f"dev{sfx}_train"), cfg["seed"])
        joblib.dump(m, out / "model_dev_all.joblib")
        for name in ("test_clean", "unlabeled_clean"):
            pr, sp = predict(m, lst(name))
            pr.to_csv(out / f"{name}.csv", index=False)
            print(name, json.dumps(sp), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="oof", choices=["oof", "final"])
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    main(a.mode, a.variant)
