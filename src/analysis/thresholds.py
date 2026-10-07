"""박스 확률 보정과 3구간 판정(통과 / 재검사 / 불량 배출) 설계 — dev OOF만 사용.

1) 보정: conf → P(TP@IoU0.5). isotonic / Platt(logit 로지스틱) 비교, fold 교차적합(3 fold 적합 → 1 fold 적용)
2) 이미지 점수: 보정 확률의 최대값. 양성 영상(전부)과 음성 대용(bg_max: GT 주변 밖 최대 점수)
3) 임계값
   t_low  = 1객체 영상에서 '위치 적중'(예측 중심이 정답 박스 안) 이미지 recall ≥ 목표(0.99)를 만족하는 가장 큰 값 → 미만은 통과
   t_high = 음성 대용 이미지 FP율 ≤ max_fpr(기본 0: dev 400장 중 0장)인 가장 작은 값 → 이상은 배출, 사이는 재검사
            (최종검사 맥락: 자동 배출은 오경보가 사실상 없을 때만. 그 외 의심은 재검사로 보냄)
4) 유병률 시나리오(0.1/1/10%)별 정밀도·재검사율·미검률 환산, 날짜 군집 부트스트랩 CI
5) --rule (2026-10-03, 근거: threshold_transfer.py, docs/decisions.md)
   max    : 위 3)의 t_low 그대로(사전 규칙, t_low_prior로도 기록)
   margin : min(t_low, p_cal(OOF F1 임계값)) — 객체를 하나라도 검출하면 자동 통과 불가
   cost   : 음성 대용 재검사 이상 비율 ≤ 0.5%를 만족하는 최소 t
   --ood_guard: dev 영상의 제품 면적 범위와 최소 띠 수(1)를 저장 → 범위 밖 영상은 통과 대신 재검사(final.py)

출력: outputs/tables/thresholds_{tag}.json, calibration_{tag}.csv, figures/calibration_{tag}.png
"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import ROOT, load_config
from src.eval import metrics as M
from src.eval.compare import load_gt


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_cal(kind, s, y):
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    return LogisticRegression(C=1e3).fit(logit(s)[:, None], y)


def apply_cal(model, kind, s):
    if kind == "isotonic":
        return model.predict(s)
    return model.predict_proba(logit(s)[:, None])[:, 1]


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2_coco")
    ap.add_argument("--target_recall", type=float, default=0.99)
    ap.add_argument("--max_fpr", type=float, default=0.0)
    ap.add_argument("--rule", choices=["max", "margin", "cost"], default="max")
    ap.add_argument("--neg_budget", type=float, default=0.005)
    ap.add_argument("--ood_guard", action=argparse.BooleanOptionalAction, default=False)
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    dev = meta[meta.split != "test"]
    ids = list(dev.index)
    preds = pd.read_csv(ROOT / "artifacts" / "preds" / a.tag / "oof.csv")
    m, n_gt = M.match(preds, gts, ids, 0.5)
    m = m.merge(preds[["image_id", "fold"]].drop_duplicates(), on="image_id")
    s, y, fold = m.score.to_numpy(), m.tp.to_numpy().astype(int), m.fold.to_numpy()
    # 교차적합 보정
    cal = {}
    for kind in ("isotonic", "platt"):
        p = np.zeros_like(s)
        for f in range(4):
            tr, te = fold != f, fold == f
            p[te] = apply_cal(fit_cal(kind, s[tr], y[tr]), kind, s[te])
        cal[kind] = dict(ece=ece(p, y), brier=float(np.mean((p - y) ** 2)), p=p)
    cal["raw"] = dict(ece=ece(s, y), brier=float(np.mean((s - y) ** 2)), p=s)
    best = min(("isotonic", "platt"), key=lambda k: cal[k]["brier"])
    m["p"] = cal[best]["p"]
    final_cal = fit_cal(best, s, y)
    joblib.dump(dict(kind=best, model=final_cal), ROOT / "artifacts" / "preds" / a.tag / "calibrator.joblib")
    # 보정 확률로 예측 표 재구성
    pc = preds.copy()
    pc["score"] = apply_cal(final_cal, best, preds.score.to_numpy())  # 표시용(전체 적합)
    pcv = preds.copy()
    sc = np.zeros(len(preds))
    for f in range(4):
        tr = fold != f
        cm = fit_cal(best, s[tr], y[tr])
        sel = (preds.fold == f).to_numpy()
        sc[sel] = apply_cal(cm, best, preds.score.to_numpy()[sel])
    pcv["score"] = sc
    hits = M.object_hits(pcv, gts, ids)
    imgs = M.image_scores(pcv, gts, ids).set_index("image_id")
    one = dev[dev.n_obj == 1].index
    # 이미지 단위 '위치 적중' 점수: 정답 박스 안에 중심이 있는 예측의 최대 확률(박스 크기 관례와 무관).
    # 영상 안의 모든 객체를 찾아야 하므로 객체별 점수의 최소값을 씀(보수적). 객체 지표(AP·F1)는 IoU 0.5 그대로.
    loc_score = hits.groupby("image_id").center_score.min()
    grid = np.unique(np.r_[np.linspace(0, 1, 1001), m.p.to_numpy(), np.nextafter(imgs.bg_max.to_numpy(), 2)])
    grid = grid[grid > 0]  # t=0이면 매칭 예측이 없는 객체(점수 0)까지 탐지로 세므로 제외
    rec1 = np.array([(loc_score.loc[one] >= t).mean() for t in grid])
    rec_all = np.array([(loc_score >= t).mean() for t in grid])
    fpr = np.array([(imgs.bg_max >= t).mean() for t in grid])
    ok_low = grid[rec1 >= a.target_recall]
    t_low = float(ok_low.max()) if len(ok_low) else 0.0  # 도달 불가 시 0 → 통과 구간 없음(전부 재검사 이상)
    target_reachable = bool(len(ok_low))
    t_low_prior = t_low
    import pandas as _pd
    cmp_ = _pd.read_csv(cfg["paths"]["tables"] / "model_compare_oof.csv")
    thr_f1 = float(cmp_[(cmp_.model == a.tag) & (cmp_.subset == "all")].thr_F1.iloc[0])
    rule_values = dict(max=t_low_prior,
                       margin=min(t_low_prior, float(apply_cal(final_cal, best, np.array([thr_f1]))[0])),
                       cost=float(grid[fpr <= a.neg_budget].min()) if (fpr <= a.neg_budget).any() else t_low_prior)
    t_low = rule_values[a.rule]
    ok_high = grid[fpr <= a.max_fpr]
    t_high = float(ok_high.min()) if len(ok_high) else 1.0
    zones = {}
    for name, sc_ in (("positive_images(one_obj)", loc_score.loc[one]), ("positive_images(all)", loc_score),
                      ("negative_proxy(bg_max)", imgs.bg_max)):
        zones[name] = dict(pass_=float((sc_ < t_low).mean()), reinspect=float(((sc_ >= t_low) & (sc_ < max(t_high, t_low))).mean()),
                           reject=float((sc_ >= max(t_high, t_low)).mean()))
    scen = []
    pos_one = loc_score.loc[one]
    for prev in (0.001, 0.01, 0.1):
        P_pass_pos = float((pos_one < t_low).mean())
        P_rej_pos = float((pos_one >= max(t_high, t_low)).mean())
        P_pass_neg = float((imgs.bg_max < t_low).mean())
        P_rej_neg = float((imgs.bg_max >= max(t_high, t_low)).mean())
        flagged_pos, flagged_neg = 1 - P_pass_pos, 1 - P_pass_neg
        scen.append(dict(prevalence=prev,
                         miss_rate_per_item=prev * P_pass_pos,
                         reinspect_rate=prev * (1 - P_pass_pos - P_rej_pos) + (1 - prev) * (1 - P_pass_neg - P_rej_neg),
                         reject_rate=prev * P_rej_pos + (1 - prev) * P_rej_neg,
                         precision_of_flag=prev * flagged_pos / max(prev * flagged_pos + (1 - prev) * flagged_neg, 1e-12)))
    # 날짜 군집 부트스트랩: t_low에서 1객체 이미지 recall
    rng = np.random.default_rng(cfg["seed"])
    d1 = dev.loc[one]
    dates = d1.date.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(dates, len(dates), replace=True)
        sel = np.concatenate([d1.index[d1.date == d].to_numpy() for d in pick])
        boots.append((loc_score.loc[sel] >= t_low).mean())
    ood = None
    if a.ood_guard:
        S = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
        ar = S.product_area.reindex(ids)
        ood = dict(area_min=float(ar.min()), area_max=float(ar.max()), min_bands=1,
                   dev_flagged=int(((ar < ar.min()) | (ar > ar.max()) | (S.n_bands.reindex(ids).fillna(0) < 1)).sum()))
    res = dict(model=a.tag, rule=a.rule, t_low_prior=t_low_prior, rule_values=rule_values, ood_guard=ood,
               calibration=best, ece={k: v["ece"] for k, v in cal.items()},
               brier={k: v["brier"] for k, v in cal.items()},
               target_recall=a.target_recall, target_reachable=target_reachable,
               max_recall_one_obj=float(rec1.max()), max_fpr=a.max_fpr, t_low=t_low, t_high=t_high,
               zone_collapsed=bool(t_high <= t_low), zones=zones,
               recall_one_obj_at_t_low=float((pos_one >= t_low).mean()),
               recall_one_obj_at_t_low_CI95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               rule_of_three_upper_miss=3 / len(one), n_one_obj=len(one), n_images=len(ids),
               negative_proxy_fpr_at_t_low=float((imgs.bg_max >= t_low).mean()), scenarios=scen)
    tab = cfg["paths"]["tables"]
    (tab / f"thresholds_{a.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame(dict(threshold=grid, recall_one_obj=rec1, recall_all=rec_all, negproxy_fpr=fpr)).to_csv(
        tab / f"threshold_curve_{a.tag}.csv", index=False)
    # 그림: 신뢰도 곡선(점 크기 = 구간 표본 수) + 3구간 경계(보정 확률, 0.75–1 확대)
    from src.report.figures_report import INK2, STATUS, style
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].plot([0, 1], [0, 1], c="#c3c2b7", lw=1, ls="--", zorder=1)
    edges = np.linspace(0, 1, 11)
    for k, c, z in (("raw", "#898781", 2), ("isotonic", "#eb6834", 3), ("platt", "#2a78d6", 4)):
        p = cal[k]["p"]
        mids, accs, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if sel.sum() >= 5:
                mids.append(p[sel].mean()); accs.append(y[sel].mean()); ns.append(sel.sum())
        ax[0].scatter(mids, accs, s=12 + 3 * np.sqrt(ns), c=c, zorder=z, label=f"{k} (ECE {cal[k]['ece']:.3f})")
    ax[0].set(xlabel="predicted P(TP) (box)", ylabel="observed TP rate", xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))
    ax[0].set_title("Box calibration (cross-fitted dev OOF; 10 bins, marker size = boxes)", fontsize=9, loc="left")
    lo_x = 0.75
    t_hi = max(t_high, t_low)
    ax[1].axvspan(lo_x, t_low, color=STATUS["pass"], alpha=0.10, lw=0)
    ax[1].axvspan(t_low, t_hi, color=STATUS["reinspect"], alpha=0.15, lw=0)
    ax[1].axvspan(t_hi, 1.0, color=STATUS["reject"], alpha=0.12, lw=0)
    ax[1].plot(grid, rec1, c="#2a78d6", label="1-object positives: localized recall")
    ax[1].plot(grid, fpr, c="#eb6834", label="negative proxy: flagged rate")
    marks = [(t_low, f"t_low = {t_low:.3f}", 0.62), (t_hi, f"t_high = {t_hi:.3f}", 0.62)]
    if abs(t_low_prior - t_low) > 1e-9:
        marks.append((t_low_prior, f"prior t_low = {t_low_prior:.3f}", 0.30))
    for t, lab, yy in marks:
        ax[1].axvline(t, ls="--", c=INK2, lw=0.8)
        ax[1].text(t - 0.004, yy, lab, rotation=90, ha="right", va="center", fontsize=7.5, color=INK2)
    for xx, lab in (((lo_x + t_low) / 2, "pass"), ((t_low + t_hi) / 2, "re-inspect")):
        ax[1].text(xx, 0.08, lab, ha="center", fontsize=8, color=INK2)
    ax[1].set(xlabel="image score = max calibrated box probability", ylabel="rate", xlim=(lo_x, 1.0), ylim=(-0.02, 1.05))
    ax[1].set_title(f"3-zone thresholds on dev OOF (rule: {a.rule}; x zoomed to {lo_x}-1)", fontsize=9, loc="left")
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    for x_ in ax:
        style(x_)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / f"calibration_{a.tag}.png", dpi=150)
    print(json.dumps({k: v for k, v in res.items() if k != "scenarios"}, indent=1))
    print(pd.DataFrame(scen).to_string(index=False))


if __name__ == "__main__":
    main()
