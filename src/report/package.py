"""제출용 소스코드 zip 생성: 스테이징 → 블라인드 점검 → MANIFEST(sha256) → zip.

사용: python -m src.report.package --data labeled|full|none [--out artifacts/package/source_code.zip]
  full    : 원본 NgImage BMP 전체 + 주 TXT 라벨 (run_all 전 단계 재현 가능, 약 0.4GB)
  labeled : 라벨 영상 500장 BMP + TXT만 (configs/subset.yaml로 실행, 미라벨 예측 재생성 불가)
  none    : 데이터 미포함 (KAMP에서 내려받아 ./data에 배치)
포함: README, requirements, run_all.py, configs/, src/, tests/, splits/, docs(일부), experiments/log.md,
      outputs/(예측·표·그림). 제외: artifacts/, reference/, CLAUDE.md, 원본 사전 문서.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config

INCLUDE = ["README.md", "requirements.txt", "run_all.py", "configs", "src", "tests", "splits",
           "docs/data_spec.md", "docs/codex_review.md", "docs/external_data.md", "docs/decisions.md",
           "experiments/log.md", "experiments/gpu_queue_d1.sh", "experiments/gpu_queue_d2.sh",
           "outputs/predictions", "outputs/tables", "outputs/figures"]


def copy(src, dst):
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["labeled", "full", "none"], default="labeled")
    ap.add_argument("--out", default="artifacts/package/source_code.zip")
    a = ap.parse_args()
    cfg = load_config()
    stage = ROOT / "artifacts" / "package" / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in INCLUDE:
        if (ROOT / rel).exists():
            copy(ROOT / rel, stage / rel)
    if a.data != "none":
        idx = pd.read_csv(cfg["paths"]["interim"] / "index.csv")
        files = idx.file if a.data == "full" else idx[idx.labeled & idx.is_canonical].file
        for f in files:
            copy(cfg["data_root"] / f, stage / "data" / f)
        copy(cfg["data_root"] / cfg["raw"]["labels_dir"], stage / "data" / cfg["raw"]["labels_dir"])
    r = subprocess.run([sys.executable, "-m", "src.report.blind_check", str(stage)], cwd=ROOT)
    if r.returncode:
        raise SystemExit("blind check failed — 패키지를 만들지 않습니다")
    man = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage / "MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    print(f"saved {out.relative_to(ROOT)}: {len(man)} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
