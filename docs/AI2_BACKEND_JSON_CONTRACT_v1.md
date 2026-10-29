# AI② → 백엔드 JSON 데이터 계약 v1 초안

이 문서는 `RealtimeRiskProcessor.process()`의 실제 반환 구조를 기준으로 한다. 성공·중복·시간 이상·입력 오류는 모두 JSON 직렬화 가능한 결과를 반환한다.

## 최상위 출력

| 필드 | 자료형 | null 가능 | 현재 의미 |
|---|---|---:|---|
| `schema_version` | string | 아니오 | 현재 `ai2-risk-evidence-v1` |
| `bridge_type` | string | 오류 시 가능 | `BOX` 또는 `TRUSS` |
| `experiment_id` | string/null | 예 | AI① 실험 식별자; HTTP ingress에서는 non-null |
| `window_id` | string | 오류 시 가능 | 입력 Window 식별자 |
| `timestamp` | string | 오류 시 가능 | 원본 timestamp 문자열 |
| `event_time` | object/null | 예 | `window_start`, `window_start_us`, `window_end_us`; HTTP ingress의 이벤트 메타데이터 |
| `message_id` | string/null | 예 | 입력의 transport metadata |
| `damage_class` | string/null | 예 | Model C 예측 클래스 |
| `damage_probabilities` | object/null | 예 | `NORMAL`, `D1`, `D2`, `D3` → number |
| `risk_score` | null | 항상 | 검증 전이므로 `null` |
| `risk_level` | null | 항상 | 검증 전이므로 `null` |
| `risk_status` | string | 아니오 | 정상 분석은 `PENDING_VALIDATION`, 오류는 `EVALUATION_ERROR` |
| `data_quality` | object | 아니오 | 상태·미검증 항목 |
| `evidence` | object | 아니오 | 기준 변화량, AI① 원본, Model C 결과 또는 오류 |
| `processing_status` | string | 아니오 | 아래 처리 상태 |
| `duplicate` | boolean | 아니오 | 중복/충돌이면 `true` |
| `out_of_order` | boolean | 아니오 | 최신 timestamp보다 늦으면 `true` |
| `received_at` | timezone 포함 ISO 8601 string | 아니오 | AI② 수신 시각 |
| `ai1_status` | string | 정상 처리 시 아니오 | `AVAILABLE` 또는 warming-up의 `PENDING_WARMUP` |

**중요:** 백엔드는 `risk_score: null`을 `0`으로, `risk_level: null`을 `NORMAL`로 변환하면 안 된다. 이는 점수가 낮다는 뜻이 아니라 아직 검증된 RiskScore 정책이 없다는 뜻이다.

## `evidence`

정상 처리의 `evidence`는 다음 객체를 포함한다.

- `baseline`: `baseline_status`, 기준 실험/Window 수, 기준 출처, 하중조건
- `feature_changes`: 22개 Feature마다 `current_value`, baseline mean/median, 절대·상대 변화량
- `ai1_result`: 원본 `ai1_result`, 보존 상태, 확인된 대표 필드
- `xgboost_model_c`: 클래스 순서, 예측 클래스, 확률, 최대 확률, Model C 식별 정보

오류 처리의 `evidence`는 `error.code`, `error.message`만 포함할 수 있다.

## `data_quality`

현재 구현된 주요 상태는 `INSUFFICIENT_BASELINE`, `OUT_OF_CONDITION`, `MISSING_REQUIRED`, `DUPLICATE_WINDOW`, `STALE_DATA`, `FUTURE_TIMESTAMP`다. `VALID`, `PROVISIONAL_BASELINE`, `SENSOR_DEGRADED`, `VERSION_MISMATCH` 구조는 지원하지만, 센서 상태·버전은 현재 확정 schema가 없어 해석하지 않는다.

HTTP ingress에서 `ai1.warming_up=true`면 `AI1_WARMING_UP` 상태와 `ai1_status=PENDING_WARMUP`을 추가한다. 이때에도 `damage_class`는 유효 Feature가 있으면 Model C의 구조 상태 가설로 반환될 수 있지만, AI① prediction을 정상/안전 상태로 해석하면 안 된다.

`not_evaluated`에는 현재 상태 저장소 또는 AI① 계약이 확정돼야 판단할 수 있는 상태가 남는다. 백엔드는 상태 배열 전체를 저장·표시해야 하며 `primary_status` 하나만으로 원인을 버리면 안 된다.

## 처리 상태 정책

| 상황 | `processing_status` | `duplicate` | `out_of_order` | 동작 |
|---|---|---:|---:|---|
| 정상 | `PROCESSED` | false | false | 평가 후 최신 교량 상태 갱신 |
| 동일 재전송 | `DUPLICATE` | true | false | 캐시 결과 반환, 재평가·재집계 안 함 |
| 동일 키 다른 내용 | `WINDOW_CONFLICT` | true | false | 오류 반환, 재평가 안 함 |
| 오래됨 | `STALE_DATA` | false | 가능 | 평가 결과는 반환하나 최신 상태 미갱신 |
| 미래 시각 | `FUTURE_TIMESTAMP` | false | 가능 | 평가 결과는 반환하나 최신 상태 미갱신 |
| 순서 역전 | `OUT_OF_ORDER` | false | true | 평가 결과는 반환하나 최신 상태 미갱신 |
| 필수 Feature 누락 | `PROCESSED` | false | false | `risk_status=EVALUATION_ERROR`, `MISSING_REQUIRED` |
| 기준 없음 | `PROCESSED` | false | false | `risk_status=EVALUATION_ERROR`, `OUT_OF_CONDITION` |

## HTTP 수신 상태 코드

| HTTP 상태 | 의미 |
|---:|---|
| 200 | 신규 분석 성공, cached duplicate, stale/future/out-of-order 분석 결과 |
| 409 | 동일 `experiment_id + window_id`의 내용 충돌 또는 bridge 충돌 |
| 413 | 설정된 요청 본문 크기 초과 |
| 422 | envelope/Feature/RAW 구조 검증 실패 |
| 503 | 모델 artifact 또는 추론 일시 불가; 내부 경로는 응답에 노출하지 않음 |
| 500 | 예상하지 못한 서버 오류; traceback은 응답에 노출하지 않음 |

현재 중복 캐시는 단일 프로세스 메모리다. 서버 재시작·다중 Uvicorn worker에서는 상태가 공유되지 않으므로 개발 서버는 단일 worker로 실행해야 한다.

## 출력 예시

[BOX 출력](../examples/ai2_to_backend_model_not_ready_box.json), [TRUSS 출력](../examples/ai2_to_backend_model_not_ready_truss.json)은 [AI① 요청 예시](AI1_AI2_JSON_CONTRACT_v1.md#실제-입력-예시)를 현재 `RealtimeRiskProcessor.process()`에 넣었을 때의 **실제 반환값**이다. 승인된 실측 Model C가 아직 없으므로 `risk_status=EVALUATION_ERROR`, `evidence.error.code=MODEL_NOT_READY`이며 `damage_class`와 `damage_probabilities`는 `null`이다. HTTP endpoint는 이 경우 결과 본문 대신 `503 MODEL_NOT_READY`를 반환한다. `tests/test_json_contract.py`가 예시 파일과 실제 반환값이 일치하는지 확인한다.

정상 분석 결과(`damage_class`, 확률, `feature_changes`, `xgboost_model_c` 포함) 예시는 승인된 실측 모델과 baseline로 실제 출력이 생긴 뒤 추가한다. 임의 수치로 만든 성공 예시는 실측 근거가 없으므로 두지 않는다.
