"""최종 평가와 제출용 예측 파일 생성. 동결 테스트셋은 이 스크립트에서 한 번만 평가합니다.

- 최종 모델: dev 전체로 학습한 {tag}_devall/last.pt (YOLO), model_dev_all.joblib (HGB 베이스라인)
- 보정기·임계값(t_low, t_high): dev OOF에서 정한 값 그대로 적용 (테스트로 조정하지 않음)
- 불확실성: 4개 fold 모델 점수의 표준편차(박스 IoU≥0.3 매칭, 없으면 0점)
- 구간 두 가지 (2026-10-03, docs/decisions.md)
    zone_prior : 사전 규칙(t_low_prior, t_high, TTA 불확실성 규칙) — 동결 테스트 1회 평가 기록
    zone       : 현행 규칙(thresholds json의 rule·t_low, + OOD 가드) — 테스트에서는 '사후 확인'으로만 보고
  --rezone   : 저장된 예측 CSV로 구간만 다시 계산 (재추론 없음, 점수·박스는 그대로)

출력
  outputs/predictions/holdout_test.csv     image_id, machine, timestamp, x1..y2(원본 px), conf, p_cal,
                                           image_score, zone, ens_mean, ens_std
  outputs/predictions/holdout_metrics.json 테스트 지표(모델별), 날짜 군집 부트스트랩 CI
  outputs/predictions/unlabeled_2032.csv   미라벨 고유 영상 추론 (정답 없음)
  outputs/predictions/{holdout_test,unlabeled_2032}_images.csv  영상 단위 판정(예측 박스가 없는 영상 포함)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analysis.thresholds import apply_cal
from src.config import ROOT, load_config
from src.eval import metrics as M
from src.eval.compare import load_gt
from src.models.predict import predict_fold, read_list, yolo_predict

Y = ROOT / "artifacts" / "yolo"


def ensemble_stats(base, fold_preds):
    """base 예측 박스마다 fold 모델별 최고 매칭 점수(IoU≥0.3) → 평균, 표준편차."""
    out = np.zeros((len(base), len(fold_preds)))
    for k, fp in enumerate(fold_preds):
        g = {i: d for i, d in fp.groupby("image_id")}
        for n, r in enumerate(base.itertuples()):
            d = g.get(r.image_id)
            if d is None:
                continue
            iou = M.iou_matrix([[r.x1, r.y1, r.x2, r.y2]], d[["x1", "y1", "x2", "y2"]].to_numpy())[0]
            sel = iou >= 0.3
            out[n, k] = d.score.to_numpy()[sel].max() if sel.any() else 0.0
    return out.mean(1), out.std(1)


TTA_U = 0.1  # dev OOF에서 검증한 재검사 규칙: 통과 구간이라도 반전 4종 보기 간 영상 점수 std ≥ 0.1이면 재검사


def tta_image_std(weights, paths, cal):
    """반전 4종 보기별 보정 영상 점수(최대 확률)의 표준편차."""
    import cv2
    from src.analysis.thresholds import apply_cal
    scores = {}
    for v, code in (("id", None), ("h", 1), ("v", 0), ("hv", -1)):
        d = Y / "images" / f"eval_final_tta_{v}"
        d.mkdir(parents=True, exist_ok=True)
        ps = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            q = d / Path(p).name
            cv2.imwrite(str(q), im if code is None else cv2.flip(im, code))
            ps.append(q)
        pr, _ = yolo_predict(weights, ps)
        pr["p"] = apply_cal(cal["model"], cal["kind"], pr.score.to_numpy()) if len(pr) else []
        scores[v] = pr.groupby("image_id").p.max()
    S = pd.DataFrame(scores).reindex([Path(p).stem for p in paths]).fillna(0.0)
    return S.std(axis=1)


def ood_flags(ids, thr, cfg):
    """OOD 가드: 제품 면적이 dev 범위 밖이거나 띠가 없는 영상(영상 구조만 사용, 표시 정보 미사용)."""
    g = thr.get("ood_guard")
    if not g:
        return pd.Series(False, index=ids)
    S = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    a, b = S.product_area.reindex(ids), S.n_bands.reindex(ids).fillna(0)
    return pd.Series(((a < g["area_min"]) | (a > g["area_max"]) | a.isna() | (b < g["min_bands"])).to_numpy(), index=ids)


def assign_zones(img, thr, ood=None, prior=False):
    """영상 단위 구간. img: index=image_id, 열 image_score(예측 없으면 0), tta_std."""
    t_low = thr.get("t_low_prior", thr["t_low"]) if prior else thr["t_low"]
    t_high = max(thr["t_high"], t_low)
    z = pd.Series(np.where(img.image_score >= t_high, "reject", np.where(img.image_score >= t_low, "reinspect", "pass")), index=img.index)
    z[(z == "pass") & (img.tta_std >= TTA_U)] = "reinspect"
    if not prior and ood is not None:
        z[(z == "pass") & ood.reindex(img.index).fillna(False)] = "reinspect"
    return z


def image_table(pr, ids, thr, ood):
    g = pr.groupby("image_id")
    img = pd.DataFrame(index=pd.Index(ids, name="image_id"))
    img["image_score"] = g.p_cal.max().reindex(ids).fillna(0.0) if len(pr) else 0.0
    img["tta_std"] = g.tta_std.first().reindex(ids).fillna(0.0) if "tta_std" in pr else 0.0
    img["ood"] = ood.reindex(ids).fillna(False)
    img["zone_prior"] = assign_zones(img, thr, prior=True)
    img["zone"] = assign_zones(img, thr, ood)
    return img


def add_decision(pr, cal, thr, meta_idx, fold_preds, tta_std=None, img=None):
    pr = pr.copy()
    pr["conf"] = pr.score
    pr["p_cal"] = apply_cal(cal["model"], cal["kind"], pr.score.to_numpy()) if len(pr) else []
    if fold_preds is not None and len(pr):
        pr["ens_mean"], pr["ens_std"] = ensemble_stats(pr, fold_preds)
    pr["image_score"] = pr.image_id.map(pr.groupby("image_id").p_cal.max())
    if tta_std is not None and len(pr):
        pr["tta_std"] = pr.image_id.map(tta_std).fillna(0.0)
    if img is not None:
        pr["zone_prior"] = pr.image_id.map(img.zone_prior)
        pr["zone"] = pr.image_id.map(img.zone)
    pr["machine"] = pr.image_id.map(meta_idx.machine)
    pr["timestamp"] = pr.image_id.map(meta_idx.timestamp)
    return pr


def zone_metrics(pr_cal, gts, ids, meta, t_low, t_high, zones):
    """영상 판정 지표(위치 적중 recall, 음성 대용, 구간 분포). zones: 영상별 구간 Series."""
    hits = M.object_hits(pr_cal.assign(score=pr_cal.p_cal), gts, ids)
    loc = hits.groupby("image_id").center_score.min()  # 이미지 판정: 위치 적중(중심이 정답 박스 안), thresholds.py와 동일
    one = [i for i in ids if meta.loc[i, "n_obj"] == 1]
    imgs = M.image_scores(pr_cal.assign(score=pr_cal.p_cal), gts, ids).set_index("image_id")
    return dict(t_low=t_low, image_recall_one_obj_at_t_low=float((loc.reindex(one).fillna(0) >= t_low).mean()),
                image_recall_all_at_t_low=float((loc.reindex(ids).fillna(0) >= t_low).mean()),
                negproxy_fpr_at_t_low=float((imgs.bg_max >= t_low).mean()),
                negproxy_fpr_at_t_high=float((imgs.bg_max >= max(t_high, t_low)).mean()),
                zones_positive=zones.reindex(ids).value_counts(normalize=True).to_dict(),
                one_obj_pass=int((zones.reindex(one) == "pass").sum()), n_one_obj=len(one),
                rule_of_three_upper_miss_one_obj=3 / max(len(one), 1))


def test_metrics(pr_raw, pr_cal, gts, ids, meta, thr, oof_thr, seed, img):
    s = M.summarize(pr_raw, gts, ids, (oof_thr,))
    m, n = M.match(pr_raw, gts, ids, 0.5)
    out = dict(images=len(ids), objects=n, AP50=s["AP50"], AP75=s["AP75"], AP50_95=s["AP50_95"],
               at_oof_F1_threshold=s["at"][0],
               R_at_FPPI_0_1=M.recall_at_fppi(pr_raw, gts, ids, 0.1)["recall"],
               **zone_metrics(pr_cal, gts, ids, meta, thr.get("t_low_prior", thr["t_low"]), thr["t_high"], img.zone_prior))
    rng = np.random.default_rng(seed)
    dates = meta.loc[ids, "date"].unique()
    obj_hit = M.object_hits(pr_raw, gts, ids).assign(hit=lambda d: d.match_score >= oof_thr)
    bt = []
    for _ in range(2000):
        pick = rng.choice(dates, len(dates), replace=True)
        sel = np.concatenate([obj_hit[obj_hit.image_id.map(meta.date) == d].hit.to_numpy() for d in pick])
        bt.append(sel.mean())
    out["object_recall_at_oof_thr_CI95"] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    out["by_machine"] = {int(k): M.summarize(pr_raw, gts, [i for i in ids if meta.loc[i, "machine"] == k], (oof_thr,))["at"][0]
                         for k in (1, 2, 3)}
    return out


def posthoc_block(pr_cal, gts, ids, meta, thr, img):
    """현행 규칙의 테스트 적용 — 규칙 설계 전에 이 테스트 결과를 보았으므로 독립 검증이 아닌 '사후 확인'."""
    if thr.get("rule", "max") == "max" and not thr.get("ood_guard"):
        return None
    return dict(note="posthoc check: the revised rule was designed after the one-shot test result was seen (not an independent validation)",
                rule=thr.get("rule"), ood_guard=thr.get("ood_guard"),
                **zone_metrics(pr_cal, gts, ids, meta, thr["t_low"], thr["t_high"], img.zone))


COLS = ["image_id", "machine", "timestamp", "x1", "y1", "x2", "y2", "conf", "p_cal", "image_score", "zone_prior", "zone",
        "tta_std", "ens_mean", "ens_std"]


def write_images(img, canon, path):
    out = img.copy()
    out.insert(0, "machine", canon.machine.reindex(img.index).to_numpy())
    out.insert(1, "timestamp", canon.timestamp.reindex(img.index).to_numpy())
    out.to_csv(path)


def rezone(a, cfg, gts, meta, canon, test_ids):
    """저장된 예측 CSV로 구간만 다시 계산합니다(재추론 없음). 기존 zone 열은 사전 규칙과 같아야 합니다."""
    tab, outdir = cfg["paths"]["tables"], cfg["paths"]["predictions"]
    thr = json.loads((tab / f"thresholds_{a.tag}.json").read_text())
    res = json.loads((outdir / "holdout_metrics.json").read_text())
    for name, ids in (("holdout_test", test_ids), ("unlabeled_2032", [p.split("/")[-1][:-4] for p in read_list("unlabeled_clean")])):
        pr = pd.read_csv(outdir / f"{name}.csv")
        old = pr.groupby("image_id").zone_prior.first() if "zone_prior" in pr else pr.groupby("image_id").zone.first()
        img = image_table(pr, ids, thr, ood_flags(ids, thr, cfg))
        mism = int((old != img.zone_prior.reindex(old.index)).sum())
        if mism:
            raise RuntimeError(f"{name}: 사전 규칙 재계산이 기존 구간과 {mism}장 다릅니다")
        pr["zone_prior"] = pr.image_id.map(img.zone_prior)
        pr["zone"] = pr.image_id.map(img.zone)
        pr[[c for c in COLS if c in pr]].to_csv(outdir / f"{name}.csv", index=False)
        write_images(img, canon, outdir / f"{name}_images.csv")
        if name == "holdout_test":
            one = [i for i in ids if meta.loc[i, "n_obj"] == 1]
            res[a.tag]["one_obj_pass"] = int((img.zone_prior.reindex(one) == "pass").sum())  # 사전 규칙(1회 평가 기록)
            pb = posthoc_block(pr, gts, ids, meta, thr, img)
            if pb:
                res[a.tag]["posthoc_revised_rule"] = pb
        else:
            res["unlabeled"]["zones_prior"] = img.zone_prior.value_counts().to_dict()
            res["unlabeled"]["zones"] = img.zone.value_counts().to_dict()
        print(name, "prior:", img.zone_prior.value_counts().to_dict(), "current:", img.zone.value_counts().to_dict())
    (outdir / "holdout_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--rezone", action="store_true", help="저장된 예측으로 구간만 재계산(재추론·재평가 없음)")
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
    canon = idx[idx.is_canonical].set_index("image_id")
    test_ids = sorted(meta[meta.split == "test"].index)
    if a.rezone:
        return rezone(a, cfg, gts, meta, canon, test_ids)
    tab = cfg["paths"]["tables"]
    outdir = cfg["paths"]["predictions"]
    outdir.mkdir(parents=True, exist_ok=True)
    cmp_ = pd.read_csv(tab / "model_compare_oof.csv")
    results = {}
    for tag in (a.tag, "hgb"):
        thr = json.loads((tab / f"thresholds_{tag}.json").read_text())
        cal = joblib.load(ROOT / "artifacts" / "preds" / tag / "calibrator.joblib")
        oof_thr = float(cmp_[(cmp_.model == tag) & (cmp_.subset == "all")].thr_F1.iloc[0])
        paths = read_list("test_clean")
        if tag == "hgb":
            from src.models import baseline_hgb as hb
            pr, sp = hb.predict(joblib.load(ROOT / "artifacts" / "preds" / "hgb" / "model_dev_all.joblib"), paths)
            folds = None
        else:
            pr, sp = yolo_predict(ROOT / "artifacts" / "runs" / f"{tag}_devall" / "weights" / "last.pt", paths)
            folds = [predict_fold(tag, f, paths)[0] for f in range(4)]
        tstd = tta_image_std(ROOT / "artifacts" / "runs" / f"{tag}_devall" / "weights" / "last.pt", paths, cal) if tag != "hgb" else None
        prc = add_decision(pr, cal, thr, canon, folds, tstd)
        img = image_table(prc, test_ids, thr, ood_flags(test_ids, thr, cfg))
        prc = add_decision(pr, cal, thr, canon, folds, tstd, img) if len(pr) else prc
        results[tag] = dict(speed=sp, thresholds=dict(t_low_prior=thr.get("t_low_prior", thr["t_low"]), t_low=thr["t_low"],
                                                      rule=thr.get("rule", "max"), t_high=thr["t_high"], oof_F1_threshold=oof_thr),
                            **test_metrics(pr, prc, gts, test_ids, meta, thr, oof_thr, cfg["seed"], img))
        pb = posthoc_block(prc, gts, test_ids, meta, thr, img)
        if pb:
            results[tag]["posthoc_revised_rule"] = pb
        if tag == a.tag:
            prc[[c for c in COLS if c in prc]].to_csv(outdir / "holdout_test.csv", index=False)
            write_images(img, canon, outdir / "holdout_test_images.csv")
            ul = read_list("unlabeled_clean")
            uid = [p.split("/")[-1][:-4] for p in ul]
            wf = ROOT / "artifacts" / "runs" / f"{tag}_devall" / "weights" / "last.pt"
            pu, spu = yolo_predict(wf, ul)
            pu = add_decision(pu, cal, thr, canon, None, tta_image_std(wf, ul, cal))
            uimg = image_table(pu, uid, thr, ood_flags(uid, thr, cfg))
            pu = add_decision(pu.drop(columns=["p_cal", "conf", "image_score", "tta_std", "machine", "timestamp"], errors="ignore"),
                              cal, thr, canon, None, uimg.tta_std, uimg)
            pu[[c for c in COLS if c in pu]].to_csv(outdir / "unlabeled_2032.csv", index=False)
            write_images(uimg, canon, outdir / "unlabeled_2032_images.csv")
            results["unlabeled"] = dict(n_images=len(ul), speed=spu, zones_prior=uimg.zone_prior.value_counts().to_dict(),
                                        zones=uimg.zone.value_counts().to_dict())
    (outdir / "holdout_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(results, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
