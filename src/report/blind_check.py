"""블라인드 점검: 제출 대상 파일에 참가자 식별 정보가 없는지 검사합니다.

검사 항목
  - 텍스트 파일 본문: 사용자명·이메일·학교명·절대경로 패턴
  - PPTX/DOCX 문서 속성(author, last_modified_by, company 등), PDF 메타데이터
  - .pt 체크포인트의 train_args 안 경로
사용: python -m src.report.blind_check <경로 또는 zip 루트> [--terms-file artifacts/blind_terms.txt] [--extra 금지어 ...]
금지어 목록은 제출물에 포함되지 않는 로컬 파일(기본 artifacts/blind_terms.txt, 한 줄에 하나)에서 읽습니다.
종료코드 1 = 위반 발견
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

# 클라우드 동기화 폴더명은 이 파일 자신이 걸리지 않도록 나눠서 조합합니다
PATH_RE = re.compile(r"(/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+|" + "One" + r"Drive)")
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".sh", ".cfg", ".ini", ".html", ".ipynb"}


def scan_text(p, terms):
    out = []
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    for t in terms:
        if t.lower() in s.lower():
            out.append(f"term '{t}'")
    for m in set(PATH_RE.findall(s)):
        out.append(f"path '{m}'")
    return out


def scan_office(p):
    out = []
    try:
        with zipfile.ZipFile(p) as z:
            for name in ("docProps/core.xml", "docProps/app.xml"):
                if name in z.namelist():
                    x = z.read(name).decode("utf-8", "ignore")
                    for tag in ("dc:creator", "cp:lastModifiedBy", "Company", "Manager"):
                        m = re.search(f"<{tag}>(.*?)</{tag}>", x)
                        if m and m.group(1).strip():
                            out.append(f"{name}:{tag}='{m.group(1).strip()}'")
    except zipfile.BadZipFile:
        out.append("bad office zip")
    return out


def scan_pdf(p):
    from pypdf import PdfReader
    meta = PdfReader(str(p)).metadata or {}
    return [f"pdf meta {k}='{v}'" for k, v in meta.items() if k in ("/Author", "/Creator", "/Producer", "/Company") and v and "Microsoft" not in str(v)]


def scan_pt(p, terms):
    import torch
    try:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        return [f"cannot load: {e}"]
    s = str(ck.get("train_args", "")) if isinstance(ck, dict) else ""
    return [f"train_args {m}" for m in set(PATH_RE.findall(s))] + [f"train_args term '{t}'" for t in terms if t in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--terms-file", default="artifacts/blind_terms.txt")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    tf = Path(a.terms_file)
    terms = ([t.strip() for t in tf.read_text(encoding="utf-8").splitlines() if t.strip()] if tf.exists() else []) + a.extra
    bad = {}
    for p in sorted(Path(a.root).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        hits = []
        if ext in TEXT_EXT:
            hits = scan_text(p, terms)
        elif ext in (".pptx", ".docx", ".xlsx"):
            hits = scan_office(p) + scan_text(p, terms)
        elif ext == ".pdf":
            hits = scan_pdf(p)
        elif ext == ".pt":
            hits = scan_pt(p, terms)
        if hits:
            bad[str(p)] = hits
    for k, v in bad.items():
        print(k, "→", "; ".join(v[:5]))
    print(f"blind check: {'FAIL' if bad else 'PASS'} ({len(bad)} files flagged)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
