"""설정 로딩과 경로 해석."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """설정 파일: 인자 > 환경변수 KMAP_CONFIG > configs/default.yaml"""
    p = Path(path or os.environ.get("KMAP_CONFIG", "configs/default.yaml"))
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_root = os.environ.get("KMAP_DATA_ROOT", cfg["data_root"])
    cfg["data_root"] = _abs(data_root)
    cfg["paths"] = {k: _abs(v) for k, v in cfg["paths"].items()}
    return cfg


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel(p):
    """보고·로그용 상대경로. 절대경로를 산출물에 남기지 않기 위해 사용합니다."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name
