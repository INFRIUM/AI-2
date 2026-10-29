# INFRIUM AI② 인수인계서

작성일: 2026-10-29 (Asia/Seoul)  
대상: AI② — 교량 손상 유형 분류 및 위험도 평가

## 1. 현재 결론

AI②는 **실측 데이터 전용 학습 및 승인 모델 전용 추론 구조**까지 구현된 상태다. 아직 실측 센서 데이터, AI① 실제 결과 파일, 승인된 baseline, 승인된 AI② 모델은 없다. 따라서 지금은 학습 또는 실시간 손상 분류가 성공하면 안 되며, 모델 미준비 상태로 차단되는 것이 정상이다.

손상 유형 클래스의 고정 순서는 다음과 같다.

```python
["NORMAL", "D1", "D2", "D3"]
```

`D1`~`D3`는 위험 등급이 아니라 손상 유형이다. `risk_score` 및 `risk_level`은 아직 실측 근거로 승인된 산식이 없으므로 생성하지 않는다.

## 2. 완료된 작업

- [x] AI② 학습 입력을 실측 데이터와 검증된 provenance로 제한했다.
- [x] BOX와 TRUSS를 별도 모델로 학습하는 XGBoost 후보 A~E 파이프라인을 구현했다.
- [x] Train으로만 학습하고 Validation으로만 후보를 비교하도록 분리했다. Test는 학습 및 후보 선택에 사용하지 않는다.
- [x] 학습 산출물은 `PENDING_VALIDATION` 상태로만 저장하며 학습만으로 실시간 추론을 허용하지 않는다.
- [x] `save_training_run()`에 실행 ID, BOX/TRUSS, A~E 아티팩트, 상태, 기존 실행 ID 충돌 사전 검증을 추가했다.
- [x] 저장본 10개(BOX/TRUSS × A~E)에 동일한 `training_run_id`를 기록하되, 메모리의 원본 아티팩트는 수정하지 않도록 했다.
- [x] baseline은 검증·승인된 실측 NORMAL 데이터만 사용하도록 제한했다.
- [x] 승인 manifest 및 SHA-256 검증을 통과한 실측 Model C만 실시간 추론에서 불러오도록 했다.
- [x] 승인 모델·baseline이 없을 때 API가 `MODEL_NOT_READY` / `EVALUATION_ERROR`로 실패하도록 처리했다.
- [x] 과거 더미 모델, 더미 baseline, 과거 분석 결과, 레거시 재학습·시각화 코드를 작업 트리에서 제거했다.

## 3. 중요한 현재 상태

### 3.1 실측 데이터가 아직 없다

다음 경로는 AI①이 실제 데이터셋으로 학습·추론한 뒤 AI②에 전달할 결과를 위한 위치이며, 현재 파일이 없는 상태가 정상이다.

```text
data/raw/measured/for_AI2/
├── box_anomaly_results.csv
├── truss_anomaly_results.csv
├── experiment_labels.csv
└── data_provenance.json

data/processed/measured/
└── ai2_experiment_split.csv
```

`src.data_loader.load_validated_data("BOX" | "TRUSS")`는 위 provenance manifest가 없거나 `data_source != "MEASURED"`, `validation_status != "VALIDATED"`이면 의도적으로 예외를 발생시킨다. 실측 파일을 준비하기 전에는 이 차단을 우회하거나 완화하면 안 된다.

### 3.2 실시간 추론은 아직 허용되지 않는다

실시간 모델 경로는 다음뿐이다.

```text
models/measured/approved/
├── approval_manifest.json
├── box/model_c.joblib
└── truss/model_c.joblib
```

현재 이 경로에는 승인 모델이 없다. `src/risk_evaluator.py`는 승인 manifest의 `MEASURED` / `VALIDATED`, 모델 메타데이터, 파일 경로 및 SHA-256을 확인한 후에만 역직렬화한다. 따라서 API는 현재 `503 MODEL_NOT_READY`를 반환해야 한다.

### 3.3 더미 자산 삭제 상태

다음 과거 자산은 삭제되었으며, Git 상태상 `D`로 표시된다. 다음 담당자는 검토 후 이 삭제를 반드시 커밋해야 한다.

- `models/box/`, `models/truss/`
- `results/baselines/`, `results/diagnostics/`, `results/risk_design/`
- `results/validation_*.csv`, `results/validation_confusion_matrices.json`
- `data/ai2_data_validation_report.csv`
- `src/recomparison.py`, 이전 모델 비교 시각화 및 더미 결과 문서/이미지

단위 테스트의 synthetic/mock 입력은 파일 기반 더미 학습 데이터가 아니다. 입력 검증, 오류 처리, 승인 검증을 확인하기 위한 메모리 내 fixture이므로 유지한다.

## 4. 핵심 파일과 역할

| 파일 | 역할 |
| --- | --- |
| `src/data_loader.py` | provenance, 라벨, 실험 단위 split을 검증하고 실측 AI① 결과를 AI② 학습용으로 로드 |
| `src/train.py` | BOX/TRUSS별 후보 A~E 학습, Validation 비교, run 단위 저장 |
| `src/baseline_manager.py` | 실측 NORMAL Train 데이터에서 baseline 후보 생성. 생성 결과는 승인 전까지 `PENDING_VALIDATION` |
| `src/feature_change.py` | 승인된 실측 baseline만 로드해 feature change 계산 |
| `src/diagnostics.py` | 특정 학습 run의 PENDING_VALIDATION 모델을 Train/Validation 범위에서 분석 |
| `src/risk_design_analysis.py` | 실측 근거 기반 위험도 산식 검토용 분석. 위험 점수 자동 확정 금지 |
| `src/risk_evaluator.py` | 승인된 실측 Model C 및 baseline으로 손상 유형 증거를 생성. RiskScore 산출 금지 |
| `src/realtime_processor.py` | 윈도우 중복·순서·상태를 관리하며 AI② 평가 호출 |
| `src/api.py` | `POST /api/ai1/windows` FastAPI endpoint |
| `docs/AI1_AI2_JSON_CONTRACT_v1.md` | AI①→AI② 전달 payload 계약 |
| `docs/AI2_BACKEND_JSON_CONTRACT_v1.md` | AI②→Backend 응답 계약 |

## 5. 학습 및 저장 계약

`src/train.py`의 `save_training_run()`은 저장 전에 다음을 모두 검증한다.

- 안전한 단일 경로 구성요소인 `run_id`인지 확인한다. 빈 값, `.`, `..`, 슬래시 및 역슬래시는 거부한다.
- BOX와 TRUSS 결과가 모두 있고, 각각 A~E 아티팩트가 모두 있는지 확인한다.
- 각 아티팩트가 dictionary이며 `data_source == "MEASURED"`, `validation_status == "PENDING_VALIDATION"`, `test_evaluated is False`인지 확인한다.
- 기존 `training_run_id`가 있으면 현재 실행 ID와 같은지 확인한다.
- 모든 검증을 마친 뒤에만 출력 경로를 만든다. 기존 run ID는 덮어쓰지 않는다.

출력 구조는 다음과 같다.

```text
models/measured/pending_validation/{run_id}/
├── box/model_a.joblib ... model_e.joblib
└── truss/model_a.joblib ... model_e.joblib

results/measured/training/{run_id}/
├── validation_model_comparison.csv
├── validation_class_metrics.csv
├── validation_confusion_matrices.json
└── training_manifest.json
```

이 산출물은 승인 모델이 아니다. 독립 Test 평가 및 승인 절차 전에는 `models/measured/approved/`에 복사하거나 API가 사용하게 하면 안 된다.

## 6. 검증 현황

마지막 실행 결과:

```text
./.venv/bin/python -m pytest tests/test_training_artifact.py -v
8 passed, 16 subtests passed

./.venv/bin/python -m pytest -q
84 passed, 1 warning, 61 subtests passed
```

경고는 FastAPI TestClient 관련 `StarletteDeprecationWarning` 1건이며, AI② 저장·승인 동작의 실패는 아니다.

유용한 명령어:

```bash
./.venv/bin/python -m py_compile src/train.py
./.venv/bin/python -m pytest tests/test_training_artifact.py -v
./.venv/bin/python -m pytest -q
git status --short
```

## 7. 다음 담당자가 수행할 우선순위

### 우선순위 0 — 변경 사항 검토 및 커밋

현재 작업 트리에는 기존 작업자가 수정한 파일과 더미 자산 삭제가 함께 존재한다. 무관한 변경을 되돌리지 말고 `git status --short`, `git diff`로 확인한 뒤 논리적인 커밋으로 나눈다. 특히 더미 자산 삭제(`D`)는 실제로 Git에 커밋해야 저장소에서 완전히 제거된다.

### 우선순위 1 — AI①·AI② 데이터 인터페이스 확정

AI① 담당자와 다음 항목을 확정하고 JSON/CSV 계약 문서를 갱신한다.

- 센서 feature 컬럼명, 단위, 결측값 표현
- window ID, event time, sample rate, window length
- `bridge_type`, `experiment_id`, 하중 조건, 온도 등 실험 메타데이터
- AI① anomaly score의 의미와 생성 split
- 라벨(`D0`→`NORMAL`, `D1`~`D3`) 및 실험 단위 ID 규칙
- 시간 동기화 및 중복 윈도우 처리 규칙

### 우선순위 2 — 실측 데이터 provenance 및 split 작성

실측 파일을 받은 뒤 `data_provenance.json`에 다음을 기록하고 독립 검토한다.

- `data_source: "MEASURED"`
- `validation_status: "VALIDATED"` (실제 검토 완료 후에만)
- `dataset_id`, `measurement_protocol`, `source_record_reference`, `validated_by`, `validated_at`
- source file 목록

Train/Validation/Test는 **실험 단위**로 분리해야 한다. 같은 `experiment_id`의 윈도우가 여러 split에 섞이면 안 된다. 또한 AI①의 학습·검증에 쓴 동일 실험이 AI② 학습 데이터에 유입되지 않도록 `ai1_split`을 검토한다.

### 우선순위 3 — 실측 baseline 생성 및 승인 검토

정상(`NORMAL`) 상태의 Train 데이터로 baseline을 생성한다.

```bash
./.venv/bin/python -m src.baseline_manager
```

생성 파일은 `results/baselines/measured/`에 생성되며 바로 승인되지 않는다. 수집 조건·실험 수·coverage를 검토하고 별도의 승인 절차가 끝난 경우에만 lookup을 `VALIDATED`로 취급한다.

### 우선순위 4 — BOX/TRUSS 실측 후보 학습

provenance 및 split 검증을 통과한 뒤 실행한다.

```bash
./.venv/bin/python -m src.train
```

실측 데이터가 없거나 provenance가 없을 때 실패하는 것은 정상이다. 성공 후 후보 선택은 Validation 지표만으로 수행하며 Test 데이터를 재선택에 사용하지 않는다.

### 우선순위 5 — 독립 Test 평가 및 모델 승인

후보 선택을 마친 뒤 고정된 후보를 독립 Test로 **한 번의 사전 정의된 평가 절차**에 따라 평가한다.

- Window 및 experiment 단위 Accuracy, Macro F1
- 클래스별 Precision, Recall, F1, Support
- Confusion Matrix
- 교량 유형별 성능
- 데이터 누수·feature leakage 재점검

평가 근거, source record, 승인자, 날짜, SHA-256을 포함한 manifest를 작성한 뒤에만 Model C를 `models/measured/approved/`로 승인한다. 평가 결과를 본 후 후보를 반복 선택하는 절차는 금지한다.

### 우선순위 6 — 위험도 산식과 통합 시험

손상 유형 분류와 위험도 등급을 분리한다. 구조·실험 근거가 확보되기 전에는 `risk_score`나 `risk_level`을 채우지 않는다. 승인 후 다음 상황을 포함해 AI①→AI②→API→Frontend를 시험한다.

- 정상 입력과 유효한 승인 모델
- 모델 또는 baseline 미준비
- 결측·잘못된 센서 입력
- 잘못된 bridge type 또는 클래스
- 중복·과거·미래 window
- 승인 manifest/해시 불일치

## 8. 절대 금지 사항

- 더미·합성 데이터를 AI② 실제 모델 학습, baseline, 성능 주장, 승인에 사용하지 않는다.
- 실측 검증 전 `VALIDATED`를 부여하지 않는다.
- `PENDING_VALIDATION` 모델을 실시간 추론에 연결하지 않는다.
- Test 데이터를 후보 선택·하이퍼파라미터 조정에 재사용하지 않는다.
- 근거 없이 `risk_score` 또는 `risk_level`을 생성하지 않는다.
- 삭제된 레거시 `models/box`, `models/truss`, `results/*` 산출물을 복원하거나 참조하지 않는다.
