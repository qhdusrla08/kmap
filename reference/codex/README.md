# ④ X-ray 검증 실험 재현 패키지

주제는 **X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석**으로 확정되었습니다. 이 패키지는 해당 데이터의 감사·기준모델·교란·오류 분석만 포함합니다.

## 빠른 시작

Python 3.12와 X-ray 원본 데이터가 필요합니다. ZIP을 새 작업 폴더에 압축 해제한 뒤 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py --data-root "C:\Users\yeon0\OneDrive\Desktop\대회\제조데이터 분석 경진대회\4. X-ray 검사장비 AI 데이터셋"
```

`--data-root`에는 **X-ray 데이터 폴더 자체**를 지정합니다. 내부에 `라벨링 6종 세트`, `NgImage`가 포함된 원래 폴더 구조가 보존되어 있어야 합니다. 원본은 수정하지 않습니다. CPU 실행이며 GPU나 사전학습 가중치는 필요하지 않습니다. 특징 추출에는 수 GB 메모리가 필요할 수 있습니다.

## 실행 과정과 생성 위치

1. `audit.py`: BMP 픽셀 중복·TXT 연결·표식·라벨 충돌 감사.
2. `extra_audit.py`: 고유 객체 단위 통계, 표식 중첩 및 전처리 예시.
3. `provenance.py`: XML 감사, 탐지용 TXT 530개의 해시, 실행 환경 기록.
4. `experiments.py`: 날짜 분할, Raw/Mask/Inpaint 기준모델, 표식 진단.
5. `robustness.py`: RF 교란 및 seed 반복.
6. `robustness_hgb.py`: HGB 교란 및 seed 반복.
7. `overlap_experiment.py`: 6월 24일의 표식 중첩군 보완 평가.
8. `summarize.py`: 후반 오류 조건, AP 정밀도, 날짜 재표집, 그래프.
9. `verify.py`: 표본 수·날짜 분리·혼동행렬·출력 일관성 검사.

재실행 수치 결과·로그·모델은 `work/`, 그래프는 `outputs/`에 생성됩니다. 단계 실패 시 실행을 중단하고 로그 위치를 출력합니다. `evidence/`는 배포 시점의 고정 결과 스냅샷이며 덮어쓰지 않습니다. `reports/`에는 보고서 HTML·Markdown과 그림이 있습니다. HTML에는 그림이 내장되어 있습니다.

재실행이 보고서의 서술을 자동으로 고치지는 않습니다. 보고서 변경과 실험 수치 변경은 별도로 검토해야 합니다. 그림은 Windows 맑은 고딕을 사용하므로 다른 운영체제에서는 `summarize.py`의 폰트 경로를 조정하십시오.

## 데이터·평가 정의

- 주요 TXT 500개와 고유 BMP 500장·1,147개 객체를 분석 집합으로 사용합니다.
- TXT 버전 충돌 3개 stem은 공식 우선순위 확인 대상입니다. 추가 TXT 12개 stem은 주 모델 비교에 포함하지 않았습니다.
- 빈 XML 또는 TXT 부재는 정상 정답을 뜻하지 않습니다.
- 주 분할은 날짜 기준 학습 294장, 검증 73장, 평가 133장입니다.
- 보완 평가의 6월 24일 104장은 별도 일자 그룹 내삽 실험입니다. 미래 날짜 예측 성능과 평균하지 않습니다.
- AP는 자체 연속 PR 적분 방식입니다. 공식 COCO 101점 AP로 표현하지 않습니다.
- 모두 양성 영상이므로 FP는 양성 영상 내부의 추가 예측 박스입니다. 정상 제품 오경보율이 아닙니다.
- 표식만 사용하는 방법은 누수 진단용입니다. 최종 탐지 모델의 성능으로 사용하지 않습니다.
- 실물 식별자가 없어 날짜 분리만으로 실물 독립성을 보장할 수 없습니다.
- 이 자료는 이미 평가 결과를 확인한 탐색·기준 실험입니다. 미접촉 최종 테스트로 간주하지 않습니다.

## 출처와 환경

`evidence/xray_inventory.csv`의 `sha`는 BMP의 RGB 픽셀 해시입니다. `evidence/source_file_manifest.json`은 탐지용 TXT의 파일 바이트 해시입니다. 원본 이미지, 배포된 가중치, 외부 데이터, 참고 문서 원문은 ZIP에 포함하지 않습니다.

`evidence/`의 절대 경로는 배포 시점의 출처 기록입니다. 재실행 시 지정한 데이터 폴더를 감사하면서 `work/`에 새로운 경로 정보를 만듭니다. 원래 사용자의 경로와 달라도 됩니다.

핵심 의존성은 `requirements.txt`, 실제 실행 환경은 `evidence/environment.json`에 기록합니다. 그래프·보조 의존성까지 고정한 설치를 원하면 `requirements-lock.txt`를 사용하십시오. Windows/Python 3.12에서 검증한 버전입니다.

## 결과 확인

`evidence/verification.json`은 논리 일관성 검사, `evidence/run_status.json`은 단계별 실행 상태, `evidence/reproduction_comparison.json`은 이전 기준 실험과의 수치 비교 결과입니다. 정리 범위와 검증 내용은 `RELEASE_NOTES.md`에 기록합니다.

검증 코드는 이번 제공 버전의 500장·1,147개 객체를 기준으로 작성했습니다. 입력 데이터가 변경되면 표본 수 검증이 실패할 수 있으므로, 원본 해시와 변경 내역을 먼저 확인하십시오.
