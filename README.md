# X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석

2026년 제6회 K-인공지능 제조데이터 분석 경진대회, 과제 ④ (X-ray 검사장비 AI 데이터셋)

## 1. 환경

- Python 3.11, CUDA 12.1 GPU에서 검증 (NVIDIA RTX 3070 8GB, 12코어 CPU, RAM 15GB, Linux/WSL2)
- 설치

```bash
conda create -n kmap python=3.11 -y
conda activate kmap
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## 2. 데이터 배치

- KAMP 「X-ray 검사장비 AI 데이터셋」 원본 폴더를 `./data`에 둡니다.
  - 기본 위치는 `./data`이고, 환경변수 `KMAP_DATA_ROOT`로 바꿀 수 있습니다.
  - 필요한 구조: `test1/yolov3/X선이물검출기(06.23_09.22)/{1,2,3}호기(…)/SN*_NgImage/*.bmp`, `라벨링 6종 세트/labels/*.txt`
- 제출 zip에 포함된 데이터가 라벨 영상 500장뿐이라면 `KMAP_CONFIG=configs/subset.yaml`로 실행합니다. 이 경우 미라벨 예측 파일은 재생성되지 않습니다.
- 원본은 읽기만 합니다. 생성물은 다음 위치에 저장됩니다.
  - `artifacts/`: 재생성 가능
  - `splits/`: 동결된 분할
  - `outputs/`: 결과
  - `experiments/logs/`: 로그

## 3. 실행 (단일 명령)

```bash
python run_all.py                    # full: 비교 모델·ablation·일반화 평가 포함 (약 8–9시간: 단계별 실측 시간 합산 추정, RTX 3070)
python run_all.py --profile core     # core: 베이스라인 + 최종모델 + 최종 평가 (약 1시간 10분: 깨끗한 디렉터리에서 실측 70.5분, RTX 3070)
python run_all.py --list             # 단계 목록
python run_all.py --from thresholds  # 특정 단계부터
python -m tests.test_metrics         # 평가 모듈 단위 테스트 (pycocotools 대조)
```

| 묶음 | 단계 | 내용 |
|---|---|---|
| 데이터 | `index`, `figures_data`, `qc_marking`, `figures_prep` | 인덱싱(픽셀 해시 중복 제거, 무결성 검사), 진단 그림, 마킹 후광 측정, 전처리 그림(제품 유형·표시 제거·가짜 표시·이식) |
| 전처리 | `preprocess`, `split`, `yolo_export`, `export_v2b`, `raw_export` | 마킹 제거(팔레트 마스크 + 2px 팽창 + telea_n), decoy·이식 학습 사본, v2b 흔적 음성(코어 마스크 삭제), 동결 분할 검증, YOLO 데이터셋 |
| 학습 | `hgb_oof`, `cv_main`, `cv_p2_v1`, `cv_p2_scratch`, `cv_p3_v1`, `cv_p2_v2b`, `raw_f3`, `devall_main`, `hgb_final` | HGB 베이스라인, **최종모델 YOLOv8s-P3 + v2b 사본**, 비교 모델(P2 v1·scratch·P3 v1·P2 v2b), 누수 대조군, 최종 학습 |
| 평가·분석 | `compare`, `thresholds_*`, `errors_*`, `leak_t1`, `leak_t2`, `leak_tests_*`, `ablation_pre`, `stress_*`, `uncertainty_main`, `generalization` | OOF 비교, 보정·3구간, 조건별 오류, 지름길 시험 T1–T8, 스트레스, 불확실성, LOMO·전향 |
| 임계값 이식 검증 | `threshold_transfer_lofo`, `nested_main`, `threshold_transfer`, `deploy_shift` | 통과 임계값 규칙 비교(LOFO·중첩 학습, 사전 고정 기준), 배포 모델 이동 |
| 3호기 대책 비교(ablation) | `export_v2bn`, `cv_p3_v2bn`, `holdout_lomo_m3_v2bn`, `noise_ablation` | 잡음 정합 증강 모델의 dev 비교(최종모델은 바꾸지 않음) |
| 결과 | `final`, `speed`, `monitoring`, `quality_drift`, `audit_pass` | 동결 테스트 단 한 번 평가, 예측 파일, 추론 속도, 미라벨 추세(top-1 점수·이물 대비), 영상 품질, 통과 영상 감사 |

## 4. 최종모델과 결과 요약

- 최종모델: **YOLOv8s-P3(COCO 사전학습) + v2b 학습 사본** (`p3_coco_v2b`, 선정 근거: `docs/decisions.md`)
- dev OOF(400장): AP50 0.969, F1 0.981, Recall@FPPI0.1 0.981
- 동결 테스트(100장, 1회 평가): AP50 0.973, P/R/F1 0.983
- 추론 약 27ms/장(전처리 포함, RTX 3070). 통과 후보 영상은 TTA(반전 3종 추가 추론)로 약 35ms가 더 들어 약 62ms/장
- 결과보고서 최종본: `reports/REPORT_최종.md`(구판 `reports/결과보고서.md`는 HWPX 반영 비교 기준으로만 보존, 변경 목록 `reports/HWPX_반영목록.md`)
- 보고서 그림 16개: `outputs/figures_report/`(`outputs/figures/`에서 복사)

## 5. 주요 출력

| 파일 | 내용 |
|---|---|
| `outputs/predictions/holdout_test.csv` | **테스트데이터 예측결과**: 동결 테스트 100장. image_id, 호기, 시각, 원본 픽셀 bbox(x1,y1,x2,y2), conf, 보정 확률, 영상 점수, 3구간 판정, TTA 표준편차, fold 앙상블 평균·표준편차(참고용, 판정에는 쓰지 않음) |
| `outputs/predictions/holdout_test_images.csv` | 영상 단위 판정(예측 박스가 없는 영상 포함): 영상 점수, TTA 표준편차, OOD 가드, `zone_prior`, `zone` |
| `outputs/predictions/holdout_metrics.json` | 테스트 지표(AP50/75/50:95, F1, Recall@FPPI, 영상 recall, 호기별, 날짜 부트스트랩 CI) |
| `outputs/predictions/unlabeled_2032.csv`, `unlabeled_2032_images.csv` | 미라벨 고유 2,032장 추론 결과(정답 없음, 박스 단위·영상 단위) |
| `outputs/tables/` | 모델 비교, 누수 시험, 스트레스, 오류 분석, 보정·임계값, 일반화, 속도 |
| `outputs/figures/` | 보고서·발표 그림(코드로 재생성). 그림 속 'Machine 1–3'은 1–3호기 |

> 판정 열: `zone_prior`는 사전 규칙(t_low 0.947, 동결 테스트 1회 평가 기록), `zone`은 현행 규칙(여유 규칙 t_low 0.833 + OOD 가드)입니다. 현행 규칙은 테스트 결과를 본 뒤 dev 시뮬레이션으로 검증해 채택했으므로, 테스트에서의 현행 규칙 결과는 '사후 확인'입니다(`holdout_metrics.json`의 `posthoc_revised_rule`, 근거 `docs/decisions.md`).

> 대회 데이터에는 공식 테스트셋이 따로 없습니다. 그래서 날짜 그룹으로 분리한 동결 hold-out(100장)을 테스트로 쓰고, 미라벨 영상의 추론 결과를 함께 제공합니다.

## 6. 장비 색상 표시 처리 요약

- **검출**: 원본은 8bit 팔레트 BMP입니다. **인덱스 244–255가 장비 표시**이며 손실 없이 정확하게 찾을 수 있습니다.
- **제거**: 테두리 바깥 약 3px까지 남는 후광을 포함하도록 **2px 팽창**한 뒤 **Telea 인페인팅 + 밝기별 잡음 합성**으로 채웁니다.
- **흔적-이물 독립화(학습 사본에만 적용)**
  - 후광까지 합성한 **decoy 표시**를 그리고 같은 방식으로 지웁니다.
  - 흔적이 없는 위치에 **Beer–Lambert 투과율로 이물을 이식**합니다.
  - (v2b) 실제 이물 30%의 코어(코어 마스크 + 3px)를 지우고 라벨을 빼서 '실제 흔적은 있고 이물은 없는' 음성을 만듭니다.
- **평가·추론 입력**은 실제 표시만 지운 영상입니다. 표시 정보는 모델 입력이나 판정 규칙에 쓰지 않습니다.
- 상세 근거: `docs/data_spec.md`, `docs/decisions.md`

## 7. 재현성

- 전역 seed 42, 영상별 난수는 `(seed, crc32(image_id), 용도)`로 고정합니다. YOLO는 `deterministic=True`로 학습하고 RAM 캐시를 끕니다.
- `splits/split_v1.json`(sha256 `fbb77e88…`)으로 동결했습니다. 재실행 시 내용이 다르면 중단합니다.
- 학습 길이는 50 epoch로 고정했습니다(early stopping 없음). OOF 평가는 `last.pt`로 하고, 임계값·보정은 dev OOF로만 정합니다.

## 8. 외부 자원

- COCO 사전학습 YOLOv8s 가중치(Ultralytics, AGPL-3.0)를 썼습니다. 사용 사유·방법·영향(scratch 비교)은 `docs/external_data.md`에 있습니다.

## 9. 데이터 출처

- 국문: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr
- 영문: Ministry of SMEs and Startups., and KAIST(Korea Advanced Institute of Science and Technology). (2020, December 14). X-ray inspection equipment AI dataset. Korea AI Manufacturing Platform(KAMP). https://www.kamp-ai.kr/
