"""보고서·발표용 요약 그림 (정적 PNG). 표에서 읽어 다시 생성합니다.

  report_leak_summary.png  지름길 시험 요약 (모델별, 지표별 작은 다중 그림)
  report_stress.png        대비 감쇠 × 위치 탐지 한계 곡선 (위치별 작은 다중 그림)
  report_zones.png         3구간 판정 분포 (양성 1객체 영상 / 음성 대용)

색: 범주형 고정 순서(검증 통과: dataviz validate_palette, light) — 모델에 고정 배정.
상태색(통과/재검사/배출)은 범주색과 분리하고 항상 글자 라벨과 함께 씁니다.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
# 비교 대상(최대 4계열, 검증된 범주색 순서 고정): 베이스라인 / v1(독립화 1단계) / P2 v2b / 최종 P3 v2b
MODELS = [("hgb", "HGB baseline"), ("p2_coco", "YOLOv8s-P2 v1"), ("p2_coco_v2b", "YOLOv8s-P2 v2b"), ("p3_coco_v2b", "YOLOv8s-P3 v2b (final)")]
COLOR = {"hgb": "#2a78d6", "p2_coco": "#eb6834", "p2_coco_v2b": "#1baf7a", "p3_coco_v2b": "#eda100"}
STATUS = {"pass": "#0ca30c", "reinspect": "#fab219", "reject": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)


def available(kind):
    out = []
    for tag, name in MODELS:
        p = TAB / (f"leak_tests_{tag}.json" if kind == "leak" else f"stress_{tag}_summary.csv" if kind == "stress" else f"thresholds_{tag}.json")
        if p.exists():
            out.append((tag, name))
    return out


def leak_summary():
    ms = available("leak")
    if not ms:
        return
    vals = {}
    for tag, _ in ms:
        L = json.loads((TAB / f"leak_tests_{tag}.json").read_text())
        st = TAB / f"stress_{tag}_summary.csv"
        floor = np.nan
        tf = np.nan
        if st.exists():
            s = pd.read_csv(st)
            floor = float(s[(s.kind == "in_place") & (s.alpha == 0.1)].recall.iloc[0]) * 100
            tf = float(s[(s.kind != "in_place") & (s.alpha == 1.0)].recall.mean()) * 100
        vals[tag] = [L["T4"]["fire_rate_erased"] * 100, (L["T5"]["recall_original"] - L["T5"]["recall_transplant"]) * 100, floor, tf]
    titles = ["T4 fire rate at erased objects (%)\nlower = no trace shortcut", "T5 recall drop, trace-free transplant (%p)\nlower = generalizes beyond traces",
              "Recall at α=0.1, original site (%)\nlower = no context shortcut", "Recall at α=1, trace-free sites (%)\nstress test, mean of 3 sites (not T5)"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    fig.patch.set_facecolor(SURF)
    x = np.arange(len(ms))
    for k, ax in enumerate(axes):
        style(ax)
        v = [vals[t][k] for t, _ in ms]
        ax.bar(x, v, width=0.7, color=[COLOR[t] for t, _ in ms], edgecolor=SURF, linewidth=2)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [n for _, n in ms], rotation=20, ha="right", fontsize=8, color=INK2)
        ax.set_title(titles[k], fontsize=9, loc="left")
        if k == 3:
            ax.set_ylim(0, 105)  # 막대는 0 기준 (축 자르기 금지)
    fig.tight_layout()
    fig.savefig(FIG / "report_leak_summary.png", dpi=170, facecolor=SURF)
    plt.close(fig)


def stress():
    ms = available("stress")
    if not ms:
        return
    kinds = [("in_place", "Original site (trace present)"), ("band_end", "Band end (no trace)"),
             ("interior", "Product interior (no trace)"), ("edge", "Product edge 5–12px (no trace)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (kind, title) in zip(axes, kinds):
        style(ax)
        for tag, name in ms:
            s = pd.read_csv(TAB / f"stress_{tag}_summary.csv")
            s = s[s.kind == kind].sort_values("alpha")
            ax.plot(s.alpha, s.recall * 100, "-o", color=COLOR[tag], lw=2, ms=4, label=name)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("contrast factor α (1 = original object)", fontsize=8, color=INK2)
        ax.invert_xaxis()
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("recall at operating threshold (%)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "report_stress.png", dpi=170, facecolor=SURF)
    plt.close(fig)


def zones():
    ms = available("thr")
    if not ms:
        return
    rows = []
    for tag, name in ms:
        T = json.loads((TAB / f"thresholds_{tag}.json").read_text())
        rule = {"margin": " [margin rule]", "cost": " [cost rule]"}.get(T.get("rule", "max"), "")
        for grp, key in (("positive (1-object)", "positive_images(one_obj)"), ("negative proxy", "negative_proxy(bg_max)")):
            z = T["zones"][key]
            rows.append((f"{name}{rule} · {grp}", z["pass_"], z["reinspect"], z["reject"]))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.2))
    fig.patch.set_facecolor(SURF)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, (zone, label) in enumerate((("pass", "pass"), ("reinspect", "re-inspect"), ("reject", "reject"))):
        w = np.array([r[1 + k] for r in rows]) * 100
        ax.barh(y, w, left=left, color=STATUS[zone], edgecolor=SURF, linewidth=2, height=0.7, label=label)
        for yi, li, wi in zip(y, left, w):
            if wi >= 6:
                txt = f"{wi:.1f}" if 99 < wi < 100 else f"{wi:.0f}"
                ax.text(li + wi / 2, yi, f"{label} {txt}%", ha="center", va="center", fontsize=7.5, color=INK)
        left += w
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of images (%) — dev OOF", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "report_zones.png", dpi=170, facecolor=SURF)
    plt.close(fig)


def main():
    leak_summary()
    stress()
    zones()
    print("report figures written:", [p.name for p in sorted(FIG.glob("report_*.png"))])


if __name__ == "__main__":
    main()
