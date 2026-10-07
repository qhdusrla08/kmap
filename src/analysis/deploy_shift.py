"""배포 모델 이동 점검 (테스트 미사용): 같은 미라벨 영상에서 fold 모델과 dev 전체 재학습 모델(devall)의 점수를 비교합니다.

3구간 임계값은 fold 모델 OOF로 정하고 devall 모델에 적용합니다. 두 모델의 점수 척도가 집단별로 다르면
임계값이 이식되지 않습니다. 정답 없이도 점수 분포는 비교할 수 있으므로 미라벨 2,032장을 씁니다.
비교 단위: 영상별 top-1 원점수(영상 판정 점수의 기반), 호기 × 띠 수(제품 유형) 집단.

출력: artifacts/preds/{tag}/unlabeled_f{f}.csv, outputs/tables/deploy_shift_{tag}.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.models.predict import predict_fold, read_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3_coco_v2b")
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "artifacts" / "preds" / a.tag
    paths = read_list("unlabeled_clean")
    ids = [p.split("/")[-1][:-4] for p in paths]
    top = {}
    for f in range(4):
        p = out / f"unlabeled_f{f}.csv"
        if not p.exists():
            pr, _ = predict_fold(a.tag, f, paths)
            pr.to_csv(p, index=False)
        top[f"f{f}"] = pd.read_csv(p).groupby("image_id").score.max()
    dv = pd.read_csv(cfg["paths"]["predictions"] / "unlabeled_2032.csv")
    top["devall"] = dv.groupby("image_id").conf.max()
    T = pd.DataFrame(top).reindex(ids).fillna(0.0)
    T["fold_mean"] = T[[f"f{f}" for f in range(4)]].mean(1)
    st = pd.read_csv(cfg["paths"]["interim"] / "structure.csv").set_index("image_id")
    T["machine"] = [int(i[0]) for i in ids]
    T["bands"] = st.n_bands.reindex(ids).fillna(0).clip(upper=3).astype(int).values
    T["product"] = np.where(T.bands >= 2, "3band", np.where(T.bands == 1, "1band", "no_band"))
    thr = json.loads((cfg["paths"]["tables"] / f"thresholds_{a.tag}.json").read_text())
    import joblib
    from src.analysis.thresholds import apply_cal
    cal = joblib.load(out / "calibrator.joblib")
    grid = np.linspace(0.001, 0.999, 999)
    pc = apply_cal(cal["model"], cal["kind"], grid)
    raw_t_low = float(grid[np.searchsorted(pc, thr.get("t_low_prior", thr["t_low"]))])
    rows = []
    for (m, prod), g in T[T.devall > 0].groupby(["machine", "product"]):
        rows.append(dict(machine=m, product=prod, n=len(g), devall_med=g.devall.median(), fold_mean_med=g.fold_mean.median(),
                         diff_med=(g.devall - g.fold_mean).median(), diff_p10=(g.devall - g.fold_mean).quantile(0.1),
                         below_raw_t_low_devall=float((g.devall < raw_t_low).mean()),
                         below_raw_t_low_folds=float(np.mean([(g[f"f{f}"] < raw_t_low).mean() for f in range(4)]))))
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["tables"] / f"deploy_shift_{a.tag}.csv", index=False)
    print(f"raw score equivalent of prior t_low: {raw_t_low:.3f}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
