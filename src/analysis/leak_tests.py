"""모델 수준 지름길 시험 T3–T7 (dev fold 모델 → 각자의 held-out fold, 테스트셋 미사용).

  T3 decoy 스트레스: clean + decoy 링(후광 합성 후 제거). decoy 자리 발화율 vs 무작위 제품 자리
  T4 이물 삭제: GT 박스 내부를 지워 '흔적은 있고 이물은 없는' 상태. 원위치 발화율 ≤5% 기대
  T5 링 없는 이식: 자기 영상의 이물을 띠 끝·제품 내부·가장자리로 이식. 원본 대비 recall
  T6 반대편 띠 끝: 이물 없는 띠 끝 자리의 점수 vs 무작위 제품 자리
  T7 차폐: 코어(5×5) 차폐 vs 링 영역(d6–13 고리) 차폐 시 점수 하락

사이트 점수 = 사이트 중심에서 체스판 거리 8px 안에 중심이 있는 예측의 최대 점수.
출력: outputs/tables/leak_tests_{tag}.json, leak_tests_{tag}_sites.csv
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import marking as mk
from src.data import structure as st
from src.data import transplant as tp
from src.data.preprocess import INWARD_RANGE, _far, decoy_centers, image_rng
from src.eval import metrics as M
from src.models.predict import predict_fold

Y = ROOT / "artifacts" / "yolo"


def site_score(preds, iid, x, y, r=8):
    p = preds[preds.image_id == iid]
    if not len(p):
        return 0.0
    cx, cy = (p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2
    sel = (np.abs(cx - x) <= r) & (np.abs(cy - y) <= r)
    return float(p.score[sel].max()) if sel.any() else 0.0


def random_product_sites(prod, avoid_pts, rng, n, d=16):
    ys, xs = np.where(prod & (st.edge_distance(prod) > 8))
    out = []
    for _ in range(200):
        if len(out) >= n or not len(xs):
            break
        k = rng.integers(len(xs))
        p = (float(xs[k]), float(ys[k]))
        if _far(p, avoid_pts, d) and _far(p, out, 14):
            out.append(p)
    return out


def ring_annulus(shape, cx, cy, d0=6, d1=13):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    cheb = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    return (cheb >= d0) & (cheb <= d1)


def build_variants(cfg, ids, canon, gt):
    """각 변형 영상을 artifacts/yolo/images/eval_{v}/에 쓰고 사이트 메타데이터를 반환."""
    out = {v: [] for v in ("decoy", "erase", "transplant", "occ_core", "occ_ring")}
    sites = []
    for v in out:
        (Y / "images" / f"eval_{v}").mkdir(parents=True, exist_ok=True)
    tgt = []  # 이식 GT
    for iid in ids:
        r = canon.loc[iid]
        clean = cv2.imread(str(Y / "images" / "clean" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        g = gt[iid]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in g]
        prod = st.product_mask(clean)
        B = st.bands(clean, prod)
        rng = image_rng(cfg["seed"], iid, 500)
        zero = np.zeros_like(clean, bool)
        # T3 decoy
        dc, kinds = decoy_centers(clean, prod, B, centers, zero, len(g), int(r.machine), rng)
        img, _, _ = mk.apply_decoys(clean, zero, dc, rng, cfg["marking"]["dilate_px"], cfg["marking"]["inpaint"])
        cv2.imwrite(str(Y / "images" / "eval_decoy" / f"{iid}.png"), img)
        sites += [dict(image_id=iid, test="T3", kind=k, x=x, y=y) for (x, y), k in zip(dc, kinds)]
        # T6 반대편 띠 끝 + 무작위 제품 자리 (clean에서 채점)
        lo, hi = INWARD_RANGE[int(r.machine)]
        for b in B:
            ax = np.array(b["axis"])
            for e in b["ends"]:
                inward = np.sign(np.dot(np.array(b["center"]) - np.array(e), ax)) * ax
                p = tuple(np.array(e) + inward * (lo + hi) / 2)
                if _far(p, centers, 16):
                    sites.append(dict(image_id=iid, test="T6", kind="band_end_empty", x=p[0], y=p[1]))
        for p in random_product_sites(prod, centers + list(dc), rng, 3):
            sites.append(dict(image_id=iid, test="T6", kind="random_product", x=p[0], y=p[1]))
        # T4 이물 삭제: GT 박스(+1px) 내부를 잡음 합성 인페인팅
        m = np.zeros_like(clean, bool)
        for b in g:
            x1, y1, x2, y2 = int(np.floor(b[0])) - 1, int(np.floor(b[1])) - 1, int(np.ceil(b[2])) + 1, int(np.ceil(b[3])) + 1
            m[max(y1, 0):y2, max(x1, 0):x2] = True
        cv2.imwrite(str(Y / "images" / "eval_erase" / f"{iid}.png"), mk.inpaint(clean, m, "telea_n", 3, rng))
        sites += [dict(image_id=iid, test="T4", kind="erased", x=c[0], y=c[1]) for c in centers]
        # T5 링 없는 이식 (원본 유지 + 이식 추가)
        img = clean.copy()
        n_t = 0
        for kind in ("band_end", "interior", "edge"):
            for _ in range(30):
                if kind == "band_end" and B:
                    b = B[rng.integers(len(B))]
                    e = b["ends"][rng.integers(2)]
                    ax = np.array(b["axis"])
                    inward = np.sign(np.dot(np.array(b["center"]) - np.array(e), ax)) * ax
                    p = tuple(np.array(e) + inward * rng.uniform(lo, hi))
                else:
                    ed = st.edge_distance(prod)
                    sel = (ed > 15) if kind == "interior" else ((ed >= 5) & (ed <= 12))
                    ys, xs = np.where(sel)
                    if not len(xs):
                        break
                    k = rng.integers(len(xs))
                    p = (float(xs[k]), float(ys[k]))
                prev = [(s["x"], s["y"]) for s in sites if s["image_id"] == iid and s["test"] == "T5"]
                if not (_far(p, centers, 18) and _far(p, prev, 18)):
                    continue
                j = int(rng.integers(len(g)))
                ex = tp.extract_ratio(clean, *centers[j])
                if ex is None or not tp.paste(img, ex[0], *p):
                    continue
                bx, by = p[0] + centers[j][0] - ex[1][0], p[1] + centers[j][1] - ex[1][1]
                w, h = g[j][2] - g[j][0], g[j][3] - g[j][1]
                tgt.append((iid, bx - w / 2, by - h / 2, bx + w / 2, by + h / 2, kind))
                sites.append(dict(image_id=iid, test="T5", kind=kind, x=bx, y=by))
                n_t += 1
                break
        cv2.imwrite(str(Y / "images" / "eval_transplant" / f"{iid}.png"), img)
        # T7 차폐
        core_img, ring_img = clean.copy(), clean.copy()
        for c in centers:
            ex = tp.extract_ratio(clean, *c)
            kx, ky = ex[1] if ex else (int(round(c[0])), int(round(c[1])))
            cm = np.zeros_like(clean, bool)
            cm[max(ky - 2, 0):ky + 3, max(kx - 2, 0):kx + 3] = True
            core_img = mk.inpaint(core_img, cm, "telea_n", 3, rng)
            ring_img = mk.inpaint(ring_img, ring_annulus(clean.shape, c[0], c[1]), "telea_n", 3, rng)
            sites.append(dict(image_id=iid, test="T7", kind="object", x=c[0], y=c[1]))
        cv2.imwrite(str(Y / "images" / "eval_occ_core" / f"{iid}.png"), core_img)
        cv2.imwrite(str(Y / "images" / "eval_occ_ring" / f"{iid}.png"), ring_img)
    return pd.DataFrame(sites), pd.DataFrame(tgt, columns=["image_id", "x1", "y1", "x2", "y2", "kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--thr", type=float, default=None, help="운영 임계값(미지정 시 OOF F1 최적값)")
    a = ap.parse_args()
    cfg = load_config()
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].set_index("image_id")
    objs = pd.read_csv(cfg["paths"]["interim"] / "objects.csv")
    gt = {i: d[["x1", "y1", "x2", "y2"]].to_numpy(float) for i, d in objs.groupby("image_id")}
    split = json.loads((cfg["paths"]["splits"] / "split_v1.json").read_text(encoding="utf-8"))["assignment"]
    thr = a.thr
    if thr is None:
        t = pd.read_csv(cfg["paths"]["tables"] / "model_compare_oof.csv")
        thr = float(t[(t.model == a.tag) & (t.subset == "all")].thr_F1.iloc[0])
    all_sites, all_tgt, preds = [], [], {v: [] for v in ("clean", "decoy", "erase", "transplant", "occ_core", "occ_ring")}
    for f in range(4):
        ids = sorted(i for i, s in split.items() if s == f"f{f}")
        s, t = build_variants(cfg, ids, canon, gt)
        all_sites.append(s.assign(fold=f)); all_tgt.append(t)
        for v in preds:
            sub = "clean" if v == "clean" else f"eval_{v}"
            pr, _ = predict_fold(a.tag, f, [Y / "images" / sub / f"{i}.png" for i in ids])
            preds[v].append(pr.assign(fold=f))
        print(f"fold {f} done", flush=True)
    sites = pd.concat(all_sites, ignore_index=True)
    tgt = pd.concat(all_tgt, ignore_index=True)
    P = {v: pd.concat(x, ignore_index=True) for v, x in preds.items()}
    pdir = ROOT / "artifacts" / "preds" / a.tag
    pdir.mkdir(parents=True, exist_ok=True)
    for v, d in P.items():
        d.to_csv(pdir / f"leak_{v}.csv", index=False)
    dev_ids = sorted(i for i, s in split.items() if s != "test")
    gts = objs[["image_id", "x1", "y1", "x2", "y2"]]

    def score_sites(df, v):
        return [site_score(P[v], r.image_id, r.x, r.y) for r in df.itertuples()]

    res = dict(model=a.tag, threshold=thr)
    # T3
    s3 = sites[sites.test == "T3"].copy()
    s3["score_decoy_img"] = score_sites(s3, "decoy")
    s3["score_clean_img"] = score_sites(s3, "clean")
    base = M.summarize(P["clean"], gts, dev_ids, (thr,))
    dec = M.summarize(P["decoy"], gts, dev_ids, (thr,))
    res["T3"] = dict(decoy_site_fire_rate=float((s3.score_decoy_img >= thr).mean()),
                     same_site_without_decoy_fire_rate=float((s3.score_clean_img >= thr).mean()),
                     n_decoys=len(s3), by_kind=s3.groupby("kind").apply(lambda d: float((d.score_decoy_img >= thr).mean())).to_dict(),
                     AP50_clean=base["AP50"], AP50_with_decoys=dec["AP50"],
                     recall_clean=base["at"][0]["recall"], recall_with_decoys=dec["at"][0]["recall"],
                     FPPI_clean=base["at"][0]["FPPI"], FPPI_with_decoys=dec["at"][0]["FPPI"])
    # T4
    s4 = sites[sites.test == "T4"].copy()
    s4["score_erased"] = score_sites(s4, "erase")
    s4["score_clean"] = score_sites(s4, "clean")
    # 합성 음성 영상(모든 이물 삭제): 영상 최대 점수 ≥ thr 비율 = 이미지 FP율의 보조 추정
    er_max = P["erase"].groupby("image_id").score.max().reindex(dev_ids).fillna(0.0)
    res["T4"] = dict(n=len(s4), fire_rate_erased=float((s4.score_erased >= thr).mean()),
                     synthetic_negative_image_fpr=float((er_max >= thr).mean()),
                     fire_rate_clean=float((s4.score_clean >= thr).mean()), pass_rule="fire_rate_erased <= 0.05",
                     passed=bool((s4.score_erased >= thr).mean() <= 0.05))
    # T5
    tg = tgt[["image_id", "x1", "y1", "x2", "y2"]]
    hits_t = M.object_hits(P["transplant"], tg, dev_ids)
    hits_t["kind"] = tgt.kind.values
    hits_o = M.object_hits(P["clean"], gts, dev_ids)
    rec = lambda h: float((h.match_score >= thr).mean())
    res["T5"] = dict(n_transplants=len(tgt), recall_original=rec(hits_o), recall_transplant=rec(hits_t),
                     recall_transplant_by_kind={k: rec(d) for k, d in hits_t.groupby("kind")},
                     n_by_kind=hits_t.kind.value_counts().to_dict(),
                     center_hit_transplant=float((hits_t.center_score >= thr).mean()),
                     pass_rule="recall_transplant >= recall_original - 0.10",
                     passed=bool(rec(hits_t) >= rec(hits_o) - 0.10))
    # T6
    s6 = sites[sites.test == "T6"].copy()
    s6["score"] = score_sites(s6, "clean")
    res["T6"] = {k: dict(n=len(d), fire_rate=float((d.score >= thr).mean()), mean_score=float(d.score.mean()))
                 for k, d in s6.groupby("kind")}
    # T7
    s7 = sites[sites.test == "T7"].copy()
    for v in ("clean", "occ_core", "occ_ring"):
        s7[f"score_{v}"] = score_sites(s7, v)
    res["T7"] = dict(n=len(s7), mean_score_clean=float(s7.score_clean.mean()),
                     mean_score_core_occluded=float(s7.score_occ_core.mean()),
                     mean_score_ring_occluded=float(s7.score_occ_ring.mean()),
                     recall_clean=float((s7.score_clean >= thr).mean()),
                     recall_core_occluded=float((s7.score_occ_core >= thr).mean()),
                     recall_ring_occluded=float((s7.score_occ_ring >= thr).mean()))
    tab = cfg["paths"]["tables"]
    (tab / f"leak_tests_{a.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat([s3, s4, s6, s7], ignore_index=True).to_csv(tab / f"leak_tests_{a.tag}_sites.csv", index=False)
    hits_t.to_csv(tab / f"leak_t5_{a.tag}_objects.csv", index=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
