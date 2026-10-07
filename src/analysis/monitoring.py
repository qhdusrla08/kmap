"""미라벨 2,032장 추론으로 본 장비별 추세 (현장 활용: 감도·판정 상태 조기경보).

가설: 영상 대부분이 4시간 주기 감도점검(시험편) 세션이므로, 같은 시험편에 대한 검출 점수와
이물 대비의 추세는 장비·판정 상태 변화를 보여 줍니다.

지표는 영상당 top-1 박스(최고 점수 박스) 하나로 계산합니다.
  - top1_conf  : top-1 원점수 (영상 판정 점수의 기반)
  - top1_depth : top-1 박스 중심의 코어 깊이(국소 배경 − 코어 최소 밝기, gray) = 영상에서 본 이물 대비
  - 구간 비율   : 통과/재검사/배출 (zone_prior = 사전 규칙, zone = 현행 규칙)
  * 2026-10-03 정정: 이전 판은 영상마다 상위 3개 박스를 섞어 집계했습니다. 1띠 제품(객체 1개)에서는
    2·3번째 박스가 배경이라 '검출 여유 급락'(3호기 08-14 19 gray, 09-21 13 gray)처럼 보였으나,
    top-1로 다시 계산하면 3호기 코어 깊이는 44–55 gray로 안정적이었습니다.
제품 유형은 영상 구조(띠 수)로 나눕니다: 3띠 / 1띠 / 띠 없음(학습 분포 밖, OOD).
출력: outputs/tables/monitoring_daily.csv, figures/monitoring_trend.png
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.conditions import core_stats
from src.config import ROOT, load_config

Y = ROOT / "artifacts" / "yolo"
MACHINE_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}  # 검증된 범주색 순서
PRODUCT_MARK = {"3band": "o", "1band": "s", "no_band": "x"}


def product_type(n_bands):
    return np.where(n_bands >= 2, "3band", np.where(n_bands == 1, "1band", "no_band"))


def image_table(cfg):
    pred = pd.read_csv(cfg["paths"]["predictions"] / "unlabeled_2032.csv", parse_dates=["timestamp"])
    ids = [p.split("/")[-1][:-4] for p in (Y / "lists" / "unlabeled_clean.txt").read_text().split()]
    idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv", parse_dates=["timestamp"])
    canon = idx[idx.is_canonical].set_index("image_id")
    st = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    top = pred.sort_values("conf", ascending=False).groupby("image_id").head(1).set_index("image_id")
    T = pd.DataFrame(index=ids)
    T["machine"] = canon.machine.reindex(ids).astype(int)
    T["timestamp"] = canon.timestamp.reindex(ids)
    T["date"] = T.timestamp.dt.date
    T["product"] = product_type(st.n_bands.reindex(ids).fillna(0).to_numpy())
    T["top1_conf"] = top.conf.reindex(ids).fillna(0.0)
    imgf = cfg["paths"]["predictions"] / "unlabeled_2032_images.csv"
    if imgf.exists():  # 영상 단위 판정(예측 박스가 없는 영상·OOD 가드 포함)
        zi = pd.read_csv(imgf, index_col="image_id")
        for z in ("zone", "zone_prior"):
            T[z] = zi[z].reindex(ids)
    else:
        for z in ("zone", "zone_prior"):
            if z in top:
                T[z] = top[z].reindex(ids).fillna("pass")  # 예측이 하나도 없는 영상은 통과
    if "zone_prior" not in T:
        T["zone_prior"] = T["zone"]
    depth = []
    for iid, r in top.iterrows():
        img = cv2.imread(str(Y / "images" / "clean" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        depth.append((iid, core_stats(img, (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2)["core_depth"]))
    T["top1_depth"] = pd.Series(dict(depth)).reindex(ids)
    T.loc[T.top1_conf < 0.1, "top1_depth"] = np.nan  # 사실상 검출이 없는 영상은 깊이 집계에서 제외
    return T


def main():
    cfg = load_config()
    T = image_table(cfg)
    rows = []
    for (m, d, prod), g in T.groupby(["machine", "date", "product"]):
        r = dict(machine=m, date=d, product=prod, n_images=len(g), top1_conf_med=g.top1_conf.median(),
                 top1_depth_med=g.top1_depth.median(), top1_depth_p10=g.top1_depth.quantile(0.1))
        for z in ("pass", "reinspect", "reject"):
            r[f"{z}_prior"] = float((g.zone_prior == z).mean())
            r[z] = float((g.zone == z).mean())
        rows.append(r)
    daily = pd.DataFrame(rows).sort_values(["machine", "date", "product"])
    daily.to_csv(cfg["paths"]["tables"] / "monitoring_daily.csv", index=False)
    T.to_csv(cfg["paths"]["tables"] / "monitoring_images.csv")
    from src.report.figures_report import INK2, style
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
    panels = (("top1_depth_med", "top-1 core depth (gray)", "Detected object contrast (top-1 box) — stable"),
              ("top1_conf_med", "top-1 raw score", "Model score scale — machine 3 drops when product switches to 1 band"),
              ("pass_prior", "share judged 'pass'", "Share of NG images judged 'pass' (pre-registered rule)"))
    for a, (col, yl, title) in zip(ax, panels):
        for m in (1, 2, 3):
            for prod, mk in PRODUCT_MARK.items():
                d = daily[(daily.machine == m) & (daily["product"] == prod) & (daily.n_images >= 3)]  # 3장 미만 묶음은 그림 제외(표에는 유지)
                if len(d):
                    a.plot(pd.to_datetime(d.date), d[col], mk, ls="", ms=4, mfc="none" if prod == "3band" else MACHINE_COLOR[m],
                           c=MACHINE_COLOR[m], label=f"Machine {m} · " + {"3band": "3-band", "1band": "1-band", "no_band": "no band"}[prod])
        a.set_ylabel(yl, color=INK2, fontsize=8)
        a.set_title(title, fontsize=9, loc="left")
        style(a)
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(0, 0.8)
    ax[2].set_ylim(0, 1.0)
    ood = daily[(daily.machine == 1) & (daily["product"] == "1band") & (pd.to_datetime(daily.date) == "2020-07-27")]
    if len(ood):  # 1호기 07-27: 학습에 없던 다른 제품(OOD) 세션 — 예측 박스가 없어 점수 0
        x0, y0 = pd.to_datetime(ood.date.iloc[0]), float(ood.top1_conf_med.iloc[0])
        ax[1].annotate("Machine 1, 07-27: different product\n(out of distribution), no box", xy=(x0, y0), xytext=(x0 + pd.Timedelta(days=4), 0.22),
                       fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(cfg["paths"]["figures"] / "monitoring_trend.png", dpi=150)
    pd.set_option("display.width", 200)
    print(T.groupby(["machine", "product"]).agg(n=("top1_conf", "size"), conf_med=("top1_conf", "median"),
                                                depth_med=("top1_depth", "median"),
                                                pass_prior=("zone_prior", lambda s: (s == "pass").mean()),
                                                pass_now=("zone", lambda s: (s == "pass").mean())).round(3).to_string())


if __name__ == "__main__":
    main()
