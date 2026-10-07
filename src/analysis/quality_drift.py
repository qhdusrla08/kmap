"""영상 품질 드리프트: 호기·일자별 영상 품질 지표와 모델 점수 척도의 관계 (현장 활용·오류분석 보조).

질문: 3호기의 점수 하락(1띠 제품 시기)이 제품 전환 때문인가, 영상 품질(잡음·밝기·대비) 변화 때문인가?
지표 (clean 영상, 고유 2,532장 — 라벨 유무 무관, 정답 미사용)
  noise_sigma     : 제품 중앙값 밝기에서의 잡음 σ (marking.noise_lut, 7×7 평균 잔차의 MAD)
  bg_level        : 제품 밖 배경 밝기 중앙값
  product_level   : 제품(띠 제외) 밝기 중앙값
  band_level      : 띠 밝기 중앙값 (띠가 없으면 결측)
  band_contrast   : product_level − band_level
  product_area, n_bands
점수: 미라벨 영상은 최종모델(devall) top-1 원점수, 라벨 dev 영상은 fold 모델 OOF top-1 원점수(참고)
분석: 호기 × 제품 유형 × 일자 중앙값 추세, 3호기 1띠 영상 안에서 점수와 지표의 순위상관,
      점수 ~ 호기×제품유형 (+ 품질 지표) 선형모형의 설명력 비교(ΔR²)
출력: outputs/tables/quality_images.csv, quality_daily.csv, quality_drift.json, figures/quality_drift.png
"""
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import ROOT, load_config
from src.data import marking as mk
from src.data import structure as st

Y = ROOT / "artifacts" / "yolo"
METRICS = ["noise_sigma", "bg_level", "product_level", "band_level", "band_contrast"]


def measure(iid):
    g = cv2.imread(str(Y / "images" / "clean" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
    prod = st.product_mask(g)
    bs = st.bands(g, prod)
    band = np.zeros(g.shape, bool)
    for b in bs:
        band |= b["mask"]
    body = prod & ~band
    lut = mk.noise_lut(g, np.zeros(g.shape, bool))
    pl = float(np.median(g[body])) if body.any() else np.nan
    bl = float(np.median(g[band])) if band.any() else np.nan
    return dict(image_id=iid, noise_sigma=float(lut[int(np.clip(pl if pl == pl else 128, 0, 255))]),
                bg_level=float(np.median(g[~prod])) if (~prod).any() else np.nan, product_level=pl, band_level=bl,
                band_contrast=pl - bl if bl == bl else np.nan)


def r2(X, y):
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var()


def main():
    cfg = load_config()
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv", parse_dates=["timestamp"])
    canon = idx[idx.is_canonical].set_index("image_id")
    ids = list(canon.index)
    with ProcessPoolExecutor(8) as ex:
        Q = pd.DataFrame(list(ex.map(measure, ids, chunksize=32))).set_index("image_id")
    S = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    Q["product_area"] = S.product_area.reindex(Q.index)
    Q["n_bands"] = S.n_bands.reindex(Q.index).fillna(0).astype(int)
    Q["product"] = np.where(Q.n_bands >= 2, "3band", np.where(Q.n_bands == 1, "1band", "no_band"))
    Q["machine"] = canon.machine.reindex(Q.index).astype(int)
    Q["date"] = canon.timestamp.reindex(Q.index).dt.date
    Q["labeled"] = canon.labeled.reindex(Q.index)
    un = pd.read_csv(cfg["paths"]["predictions"] / "unlabeled_2032.csv").groupby("image_id").conf.max()
    oof = pd.read_csv(ROOT / "artifacts" / "preds" / "p3_coco_v2b" / "oof.csv").groupby("image_id").score.max()
    Q["top1_devall"] = un.reindex(Q.index)
    Q.loc[~Q.labeled, "top1_devall"] = Q.loc[~Q.labeled, "top1_devall"].fillna(0.0)
    Q["top1_oof"] = oof.reindex(Q.index)
    tab = cfg["paths"]["tables"]
    Q.to_csv(tab / "quality_images.csv")
    daily = Q.groupby(["machine", "date", "product"])[METRICS + ["top1_devall"]].median()
    daily["n"] = Q.groupby(["machine", "date", "product"]).size()
    daily.reset_index().to_csv(tab / "quality_daily.csv", index=False)
    res = {}
    res["by_machine_product"] = json.loads(Q.groupby(["machine", "product"])[METRICS].median().round(3).reset_index().to_json(orient="records"))
    # 3호기 1띠(미라벨) 안에서 점수와 품질 지표의 순위상관
    m31 = Q[(Q.machine == 3) & (Q["product"] == "1band") & ~Q.labeled & (Q.top1_devall > 0.1)]
    res["m3_1band_spearman"] = {c: dict(rho=float(stats.spearmanr(m31[c], m31.top1_devall, nan_policy="omit")[0]),
                                        p=float(stats.spearmanr(m31[c], m31.top1_devall, nan_policy="omit")[1]))
                                for c in METRICS}
    res["m3_1band_n"] = len(m31)
    # 점수 설명력: 호기×제품유형 더미 vs + 품질 지표 (미라벨, 띠 있는 영상, 검출 있는 영상)
    D = Q[~Q.labeled & (Q["product"] != "no_band") & (Q.top1_devall > 0.1)].dropna(subset=["noise_sigma", "product_level", "bg_level"])
    grp = pd.get_dummies(D.machine.astype(str) + "_" + D["product"], drop_first=True).to_numpy(float)
    qual = D[["noise_sigma", "product_level", "bg_level"]].to_numpy(float)
    qual = (qual - qual.mean(0)) / qual.std(0)
    y = D.top1_devall.to_numpy()
    res["r2_group"] = float(r2(grp, y))
    res["r2_quality"] = float(r2(qual, y))
    res["r2_group_plus_quality"] = float(r2(np.c_[grp, qual], y))
    res["n_regression"] = len(D)
    (tab / "quality_drift.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    # 그림: 잡음 σ, 제품 밝기, 띠 대비의 호기·일자 추세
    from src.analysis.monitoring import MACHINE_COLOR, PRODUCT_MARK
    from src.report.figures_report import INK2, style
    dd = daily.reset_index()
    dd = dd[dd.n >= 3]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for a, (col, yl) in zip(ax, (("noise_sigma", "noise σ at product level"), ("product_level", "product brightness (gray)"),
                                 ("band_contrast", "band contrast (gray)"))):
        for m in (1, 2, 3):
            for prod, mk_ in PRODUCT_MARK.items():
                d = dd[(dd.machine == m) & (dd["product"] == prod)]
                if len(d):
                    a.plot(pd.to_datetime(d.date), d[col], mk_, ls="", ms=4, c=MACHINE_COLOR[m],
                           mfc="none" if prod == "3band" else MACHINE_COLOR[m], label=f"Machine {m} · " + {"3band": "3-band", "1band": "1-band", "no_band": "no band"}[prod])
        a.set_ylabel(yl, color=INK2, fontsize=8)
        a.set_ylim(bottom=0)
        style(a)
    ax[0].set_title("Image quality by machine and day (unique images, clean)", fontsize=9, loc="left")
    ood = dd[(dd.machine == 1) & (dd["product"] == "1band") & (pd.to_datetime(dd.date) == "2020-07-27")]
    if len(ood):  # 1호기 07-27: 다른 제품(OOD) 세션 영상이 섞인 묶음
        x0, y0 = pd.to_datetime(ood.date.iloc[0]), float(ood.band_contrast.iloc[0])
        ax[2].annotate("Machine 1, 07-27: different product\n(out of distribution) mixed in", xy=(x0, y0), xytext=(x0 + pd.Timedelta(days=4), 8),
                       fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(cfg["paths"]["figures"] / "quality_drift.png", dpi=150)
    pd.set_option("display.width", 200)
    print(Q.groupby(["machine", "product"])[METRICS + ["top1_devall"]].median().round(2).to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "by_machine_product"}, indent=1))


if __name__ == "__main__":
    main()
