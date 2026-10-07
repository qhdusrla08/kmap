"""3구간 '통과' 임계값(t_low)의 이식성 검증 — dev 안 시뮬레이션만 사용 (테스트 미사용).

배경: 사전 규칙(R0: 1객체 영상 위치 적중 recall ≥ 0.99를 만족하는 최대 t)을 dev OOF에서 정해 dev 전체 재학습
모델에 적용하자 동결 테스트에서 1객체 양성 61장 중 9장(전부 3호기 1띠)이 통과로 판정됐습니다.
아래 규칙 후보는 그 결과를 본 뒤 설계했으므로, 비교·선택은 실행 전에 고정한 기준으로 dev에서만 합니다.

규칙 (모두 '안쪽' 예측만으로 정하고 '바깥' 예측에 적용, 원점수 공간에서 계산 — 보정은 단조이므로 판정 동일)
  R0 사전    : 안쪽 1객체 영상 위치 적중 점수(객체별 중심 적중 점수의 최소값)의 recall ≥ 0.99를 만족하는 최대 t
  R1 여유    : min(R0, 안쪽 OOF의 객체 F1 최적 임계값) → '객체를 하나라도 검출하면 자동 통과 불가'
  R2 호기별  : 호기마다 R0 (그 호기의 안쪽 1객체 영상이 없으면 전체 R0)
  R3 보수    : 날짜 군집 부트스트랩(2,000회) R0 분포의 5% 분위수
  R4 비용상한: 안쪽 음성 대용(bg_max)의 재검사 이상 비율 ≤ 0.5%를 만족하는 최소 t
               (R0의 쌍대: R0는 recall 제약 아래 t를 최대화해 양성 쪽 여유가 0, R4는 비용 제약 아래 t를 최소화해
                양성 쪽 여유를 최대화. 예산 0.5%는 합격 기준 1%의 절반 — recall 쪽 0.99 설계와 같은 이중 여유)
               [2026-10-03 추가: LOFO에서 R0–R3가 모두 같은 누수(1객체 2장)를 보인 뒤, 중첩 결과가 나오기 전에 고정]
  OOD 가드   : 제품 면적이 안쪽 영상 범위 밖이거나 띠가 0개인 영상은 통과 대신 재검사 (영상 구조만 사용)
측정 (바깥)
  pass_one / pass_all : 통과로 판정된 1객체 / 전체 양성 영상 수 (영상 최대 원점수 < t, OOD 가드 제외)
  neg_flag            : 음성 대용(bg_max) 영상 중 t 이상(재검사 이상)인 비율 = 과검 비용
모드
  lofo   : 바깥 fold k, 안쪽 = 나머지 3 fold의 기존 OOF (집단·날짜 이동만 반영)
  nested : 바깥 k ∈ {2,3}, 안쪽 = 2-fold 학습 안쪽 모델의 OOF(src.models.nested), 배포 = 기존 fold 모델 f_k
           (배포 모델 이동까지 반영). 부가 후보 'ens': 안쪽 모델 3개의 바깥 영상 점수 평균을 배포로 사용
사전 고정 선택 기준
  ① 두 모드 모두 neg_flag ≤ 1% ② 그중 1객체 누수 합 → 전체 누수 합이 가장 적은 규칙
  ③ 동률이면 R0(현행 유지) → R1 → R4 → R3 → R2 순
출력: outputs/tables/threshold_transfer_{tag}.csv, threshold_transfer_{tag}.json, deploy_shift_dev_{tag}.csv,
      outputs/figures/threshold_tradeoff_{tag}.png (dev OOF: t_low 위치별 양성 통과 누수 vs 음성 대용 재검사 비용)
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.eval import metrics as M
from src.eval.compare import best_f1, load_gt

RULES = ("R0", "R1", "R2", "R3", "R4")
TIE_ORDER = ("R0", "R1", "R4", "R3", "R2")
TARGET = 0.99
MAX_NEG_FLAG = 0.01
NEG_BUDGET = 0.005


def r0_threshold(loc):
    """recall(loc ≥ t) ≥ TARGET를 만족하는 최대 t. 위치 적중이 없는 객체(0점)는 t>0에서 항상 미검."""
    s = np.sort(np.asarray(loc, float))
    if not len(s):
        return None
    allowed = int(np.floor((1 - TARGET) * len(s) + 1e-9))
    t = s[allowed]
    return float(t) if t > 0 else 1e-6


def ood_flags(ids, st, lo, hi):
    a = st.product_area.reindex(ids)
    b = st.n_bands.reindex(ids).fillna(0)
    return pd.Series(((a < lo) | (a > hi) | a.isna() | (b == 0)).to_numpy(), index=ids)


def select_rules(P_in, ids_in, gts, meta, seed):
    """안쪽 예측(원점수)으로 규칙별 임계값을 정합니다. 반환: {rule: t 또는 {machine: t}}."""
    hits = M.object_hits(P_in, gts, ids_in)
    loc = hits.groupby("image_id").center_score.min()
    one = [i for i in ids_in if meta.loc[i, "n_obj"] == 1]
    t0 = r0_threshold(loc.reindex(one).fillna(0))
    m, n = M.match(P_in, gts, ids_in, 0.5)
    thr_f1 = float(best_f1(m, n, len(ids_in))["threshold"])
    out = {"R0": t0, "R1": min(t0, thr_f1), "thr_F1": thr_f1}
    out["R2"] = {}
    for mc in (1, 2, 3):
        om = [i for i in one if meta.loc[i, "machine"] == mc]
        out["R2"][mc] = r0_threshold(loc.reindex(om).fillna(0)) if len(om) >= 10 else t0
    rng = np.random.default_rng(seed)
    d1 = meta.loc[one]
    dates = d1.date.unique()
    by_date = {d: loc.reindex(d1.index[d1.date == d]).fillna(0).to_numpy() for d in dates}
    boots = [r0_threshold(np.concatenate([by_date[d] for d in rng.choice(dates, len(dates), replace=True)]))
             for _ in range(2000)]
    out["R3"] = float(np.percentile(boots, 5))
    bg = np.sort(M.image_scores(P_in, gts, ids_in).bg_max.to_numpy())[::-1]
    allowed = int(np.floor(NEG_BUDGET * len(bg) + 1e-9))
    out["R4"] = max(float(np.nextafter(bg[allowed], 2.0)), 1e-6)  # bg ≥ t 인 영상이 allowed장 이하
    return out


def evaluate(top, bg, ids_out, meta, ood, rules, label):
    """바깥 영상에 규칙 적용. top/bg: 영상별 최대 원점수 / 음성 대용 점수(Series)."""
    rows = []
    one = [i for i in ids_out if meta.loc[i, "n_obj"] == 1]
    for r in RULES:
        t = rules[r]
        tv = pd.Series([t[int(meta.loc[i, "machine"])] if isinstance(t, dict) else t for i in ids_out], index=ids_out)
        passed = (top.reindex(ids_out).fillna(0) < tv) & ~ood.reindex(ids_out).fillna(False)
        m3one = [i for i in one if meta.loc[i, "machine"] == 3]
        rows.append(dict(**label, rule=r, t_raw=(json.dumps({k: round(v, 4) for k, v in t.items()}) if isinstance(t, dict) else round(t, 4)),
                         n_img=len(ids_out), pass_all=int(passed.sum()), n_one=len(one), pass_one=int(passed[one].sum()),
                         n_m3_one=len(m3one), pass_m3_one=int(passed[m3one].sum()),
                         n_neg=len(ids_out), neg_flag=int((bg.reindex(ids_out).fillna(0) >= tv).sum()),
                         ood_flagged=int(ood.reindex(ids_out).fillna(False).sum())))
    return rows


def image_stats(P, gts, ids):
    s = M.image_scores(P, gts, ids).set_index("image_id")
    return s.image_score, s.bg_max


def tradeoff_figure(oof, gts, meta, dev, chosen_t, cfg, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.report.figures_report import INK2, STATUS, style
    top, bg = image_stats(oof, gts, dev)
    one = [i for i in dev if meta.loc[i, "n_obj"] == 1]
    m3 = [i for i in one if meta.loc[i, "machine"] == 3]
    grid = np.logspace(-3, 0, 400)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(grid, [(top.reindex(one).fillna(0) < t).mean() for t in grid], c="#2a78d6", label="positive 1-object images passed (all machines)")
    ax.plot(grid, [(top.reindex(m3).fillna(0) < t).mean() for t in grid], c="#1baf7a", ls="--", label="positive 1-object images passed (machine 3)")
    ax.plot(grid, [(bg.reindex(dev).fillna(0) >= t).mean() for t in grid], c="#eb6834", label="negative-proxy images flagged (reinspect cost)")
    for r, c in (("R0", STATUS["reject"]), ("R4", STATUS["pass"])):
        v = [d[r] for d in chosen_t.values() if r in d and not isinstance(d[r], dict)]
        if v:
            ax.axvspan(min(v), max(v), color=c, alpha=0.15, lw=0)
            ax.text(np.sqrt(min(v) * max(v)), 0.92, r, ha="center", fontsize=8, color=INK2)
    thr_p = cfg["paths"]["tables"] / f"thresholds_{tag}.json"
    cal_p = ROOT / "artifacts" / "preds" / tag / "calibrator.joblib"
    if thr_p.exists() and cal_p.exists():  # 운영 임계값(보정 확률)을 원점수로 환산해 표시
        import joblib
        from src.analysis.thresholds import apply_cal
        T, C = json.loads(thr_p.read_text()), joblib.load(cal_p)
        g = np.linspace(1e-4, 0.9999, 20000)
        pc = apply_cal(C["model"], C["kind"], g)
        raw = lambda t: float(g[min(np.searchsorted(pc, t), len(g) - 1)])
        tp = T.get("t_low_prior", T["t_low"])
        lines = [(raw(tp), f"prior t_low\nraw {raw(tp):.3f} = p_cal {tp:.3f}", 0.45)]
        if T.get("rule", "max") != "max":
            lines.append((raw(T["t_low"]), f"final t_low (R1, {T['rule']} rule)\nraw {raw(T['t_low']):.3f} = p_cal {T['t_low']:.3f}", 0.45))
        for x, lab, yy in lines:
            ax.axvline(x, c=INK2, ls="--", lw=0.8)
            ax.text(x * 0.97, yy, lab, rotation=90, ha="right", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="t_low (raw score, log scale)", ylabel="rate", ylim=(0, 1))
    ax.set_title("Pass threshold trade-off (dev OOF). Shaded: thresholds chosen by each rule across simulation splits", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="center left")
    style(ax)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / f"threshold_tradeoff_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3_coco_v2b")
    ap.add_argument("--modes", default="lofo,nested")
    a = ap.parse_args()
    cfg = load_config()
    gts, meta = load_gt(cfg)
    st = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    pdir = ROOT / "artifacts" / "preds" / a.tag
    oof = pd.read_csv(pdir / "oof.csv")
    dev = [i for i in meta.index if meta.loc[i, "split"] != "test"]
    fold_of = pd.Series({i: int(meta.loc[i, "split"][1]) for i in dev})
    rows, shift_rows, chosen_t = [], [], {}
    modes = a.modes.split(",")
    for mode in modes:
        outers = range(4) if mode == "lofo" else (2, 3)
        for k in outers:
            ids_in = [i for i in dev if fold_of[i] != k]
            ids_out = [i for i in dev if fold_of[i] == k]
            if mode == "lofo":
                P_in = oof[oof.fold != k]
            else:
                parts = [pd.read_csv(pdir / f"nested_k{k}_j{j}.csv") for j in range(4) if j != k
                         if (pdir / f"nested_k{k}_j{j}.csv").exists()]
                if len(parts) < 3:
                    print(f"nested k={k}: 안쪽 예측 {len(parts)}/3 → 건너뜀")
                    continue
                P_in = pd.concat([p[p.part == "inner"] for p in parts])
            rules = select_rules(P_in, ids_in, gts, meta, cfg["seed"])
            a_in = st.product_area.reindex(ids_in)
            ood = ood_flags(ids_out, st, a_in.min(), a_in.max())
            top, bg = image_stats(oof[oof.fold == k], gts, ids_out)  # 배포 = 기존 fold 모델 f_k
            label = dict(mode=mode, outer=k, deploy="fold_model")
            rows += evaluate(top, bg, ids_out, meta, ood, rules, label)
            chosen_t[f"{mode}_k{k}"] = {r: rules[r] for r in RULES} | {"thr_F1": rules["thr_F1"]}
            if mode == "nested":
                outs = [p[p.part == "outer"] for p in parts]
                st_out = [image_stats(o, gts, ids_out) for o in outs]
                top_e = pd.concat([s[0] for s in st_out], axis=1).mean(1)
                bg_e = pd.concat([s[1] for s in st_out], axis=1).mean(1)
                rows += evaluate(top_e, bg_e, ids_out, meta, ood, rules, dict(mode=mode, outer=k, deploy="inner_ensemble"))
                # 배포 모델 이동: 같은 바깥 객체에서 배포(f_k) − 안쪽 모델 평균의 위치 적중 점수
                hd = M.object_hits(oof[oof.fold == k], gts, ids_out).set_index(["image_id", "obj"]).center_score
                hi = pd.concat([M.object_hits(o, gts, ids_out).set_index(["image_id", "obj"]).center_score for o in outs], axis=1).mean(1)
                d = pd.DataFrame(dict(deploy=hd, inner=hi)).reset_index()
                d["machine"] = d.image_id.map(meta.machine)
                d["n_obj"] = d.image_id.map(meta.n_obj)
                for (mc, no), g in d.groupby(["machine", "n_obj"]):
                    shift_rows.append(dict(outer=k, machine=mc, n_obj=no, n=len(g), deploy_med=g.deploy.median(),
                                           inner_med=g.inner.median(), diff_med=(g.deploy - g.inner).median(),
                                           deploy_p05=g.deploy.quantile(.05), inner_p05=g.inner.quantile(.05)))
    df = pd.DataFrame(rows)
    tab = cfg["paths"]["tables"]
    df.to_csv(tab / f"threshold_transfer_{a.tag}.csv", index=False)
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(tab / f"deploy_shift_dev_{a.tag}.csv", index=False)
    main_rows = df[df.deploy == "fold_model"]
    pooled = main_rows.groupby(["mode", "rule"])[["n_img", "pass_all", "n_one", "pass_one", "n_m3_one", "pass_m3_one", "n_neg", "neg_flag"]].sum()
    pooled["neg_flag_rate"] = pooled.neg_flag / pooled.n_neg
    from scipy.stats import beta
    # 1객체 누수율의 단측 95% 상한(Clopper–Pearson). 0건이면 rule of three(≈3/n)와 같음
    pooled["one_leak_upper95"] = [float(beta.ppf(0.95, k + 1, n - k)) for k, n in zip(pooled.pass_one, pooled.n_one)]
    ok = {r: all(pooled.loc[(m, r), "neg_flag_rate"] <= MAX_NEG_FLAG for m in pooled.index.get_level_values(0).unique())
          for r in RULES}
    tot = main_rows.groupby("rule")[["pass_one", "pass_all"]].sum()
    cand = [r for r in TIE_ORDER if ok[r]]
    selected = min(cand, key=lambda r: (tot.loc[r, "pass_one"], tot.loc[r, "pass_all"], TIE_ORDER.index(r))) if cand else None
    res = dict(tag=a.tag, modes=modes, criteria=dict(target_recall=TARGET, max_neg_flag_rate=MAX_NEG_FLAG,
                                                    tie_order=TIE_ORDER, note="실행 전 고정, dev만 사용"),
               cost_ok=ok, leak_totals=tot.to_dict(), selected_rule=selected,
               pooled=json.loads(pooled.reset_index().to_json(orient="records")), thresholds_by_split=chosen_t)
    (tab / f"threshold_transfer_{a.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    tradeoff_figure(oof, gts, meta, dev, chosen_t, cfg, a.tag)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    print(pooled.round(4).to_string())
    print("cost ok:", ok, "→ selected:", selected)
    if shift_rows:
        print(pd.DataFrame(shift_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
