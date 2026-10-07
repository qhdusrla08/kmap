"""3호기 대책 모델(잡음 정합 증강 v2bn) dev 비교 — ablation 전용, 최종모델 교체 없음 (docs/decisions.md 2026-10-03).

같은 분할·같은 평가 모듈로 최종모델(p3_coco_v2b)과 비교합니다. 테스트셋은 쓰지 않습니다.
  - dev OOF: 전체·3호기 (사전 선정 기준 순서 R@FPPI0.1 → F1 → AP50:95)
  - 3호기 1객체 영상 top-1 점수 분포 (점수 척도: 통과 임계값 이식 실패의 원인)
  - 호기 hold-out 3호기(1·2호기로 학습 → 3호기 평가)
  - 3구간 통과 임계값 LOFO 누수 (threshold_transfer_{tag}.json)
  - 지름길 점검 T4(이물 삭제 발화율), T7(코어 차폐 recall)
출력: outputs/tables/noise_ablation.csv
"""
import json

import pandas as pd

from src.config import ROOT, load_config
from src.eval.compare import load_gt
from src.eval.generalization import row
from src.models import baseline_hgb as hb

TAGS = [("p3_coco_v2b", "final (v2b)"), ("p3_coco_v2bn", "noise-matched (v2bn)")]


def main():
    cfg = load_config()
    gts, meta = load_gt(cfg)
    tab = cfg["paths"]["tables"]
    cmp_ = pd.read_csv(tab / "model_compare_oof.csv")
    lomo_ids = [l.split("/")[-1][:-4] for l in hb.lst("lomo_m3_val")]
    rows = []
    for tag, name in TAGS:
        c = cmp_[cmp_.model == tag].set_index("subset")
        if c.empty:
            print(f"{tag}: OOF 비교 행 없음 → 건너뜀")
            continue
        r = dict(model=tag, name=name)
        for sub in ("all", "machine3"):
            for k in ("AP50", "AP50_95", "R_at_FPPI_0_1", "F1_best", "thr_F1"):
                r[f"{sub}_{k}"] = float(c.loc[sub, k])
        oof = pd.read_csv(ROOT / "artifacts" / "preds" / tag / "oof.csv").groupby("image_id").score.max()
        m31 = [i for i in meta.index if meta.loc[i, "split"] != "test" and meta.loc[i, "machine"] == 3 and meta.loc[i, "n_obj"] == 1]
        t = oof.reindex(m31).fillna(0)
        r.update(m3_one_top1_median=float(t.median()), m3_one_top1_p05=float(t.quantile(0.05)), m3_one_top1_min=float(t.min()))
        p = ROOT / "artifacts" / "preds" / tag / "lomo_m3.csv"
        if p.exists():
            g = row(tag, "lomo_m3", pd.read_csv(p), gts, lomo_ids, r["all_thr_F1"])
            r.update(lomo_m3_AP50=g["AP50"], lomo_m3_F1=g["F1"], lomo_m3_R_at_FPPI=g["R_at_FPPI_0_1"])
        tt = tab / f"threshold_transfer_{tag}.json"
        if tt.exists():
            pooled = {(d["mode"], d["rule"]): d for d in json.loads(tt.read_text())["pooled"]}
            for rule in ("R0", "R1"):
                d = pooled.get(("lofo", rule))
                if d:
                    r[f"lofo_{rule}_one_obj_leak"] = d["pass_one"]
                    r[f"lofo_{rule}_negproxy_cost"] = d["neg_flag_rate"]
        lk = tab / f"leak_tests_{tag}.json"
        if lk.exists():
            L = json.loads(lk.read_text())
            r.update(T4_fire_erased=L["T4"]["fire_rate_erased"], T7_recall_core_occluded=L["T7"]["recall_core_occluded"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(tab / "noise_ablation.csv", index=False)
    print(df.set_index("name").T.round(3).to_string())


if __name__ == "__main__":
    main()
