# AI① → AI② JSON 데이터 계약 v1 초안

## 범위와 구현 기준

이 문서는 현재 `feature_change.py`, `risk_evaluator.py`, `realtime_processor.py`와 FastAPI 수신 endpoint `POST /api/ai1/windows` 구현을 기준으로 한다. WebSocket, 영속 저장소, Django 전달은 아직 구현되지 않았다. `SYNTHETIC` 예시는 실제 HW 측정값이 아니다.

## FastAPI 요청 envelope (STEP 6-8-2)

AI①은 `Content-Type: application/json`으로 아래 중첩 구조를 전송한다. HTTP 요청에서는 `experiment_id`, `message_type`, `window_start_us`, `window_end_us`, `raw`, `ai1`이 필수다. 개발 기본 본문 제한은 1 MiB이며 `AI2_MAX_REQUEST_BYTES` 환경변수로만 조정한다.

| HTTP 필드 | 자료형/검증 | AI② 내부 변환 |
|---|---|---|
| `schema_version` | 비어 있지 않은 string | canonical `schema_version` |
| `message_type` | 정확히 `ai1_window` | HTTP envelope 전용 |
| `bridge_type` | `BOX` 또는 `TRUSS` | canonical `bridge_type` |
| `experiment_id` | 비어 있지 않은 string | 중복 키와 출력에 보존 |
| `window_id` | 비어 있지 않은 string | 중복 키와 출력에 보존 |
| `window_start` | string, 원문 보존 | `event_time.window_start` |
| `window_start_us`, `window_end_us` | integer Unix microseconds, start < end | UTC event time/메타데이터 |
| `raw` | object 또는 null | 수신·구조 검증 후 Model C/cache에는 미복사 |
| `ai1` | object | Feature·환경·AI① 근거로 변환 |

`raw.sensors.<sensor>`는 null일 수 있다. 센서 객체에 알려진 `sample_rate_hz`, `unit`, `t0_us`, `n`, `axes`, 선택 `timestamp_us`, `quality`가 있으면 검증하며, `axes`/`timestamp_us` 배열이 존재할 때 길이는 `n`과 일치해야 한다. 원본 API_SPEC.md는 현재 프로젝트에 없으므로 그 외 RAW 센서 세부 필드는 추가 확인이 필요하다.

시간대 없는 `window_start`는 권위 있는 시간으로 사용하지 않는다. AI②는 `window_end_us`를 UTC timezone-aware ISO 8601 `timestamp`로 변환하며, `received_at`은 별도의 서버 수신 시각이다.

## AI① → canonical AI② 변환

| AI① HTTP 위치 | canonical AI② 위치 | 비고 |
|---|---|---|
| `ai1.features` | `features` | 아래 22개만 이름 기반으로 Model C 순서 재구성 |
| `ai1.condition.load_level_kg` | `environment.load_level_kg` | baseline 조회 사용 |
| `ai1.condition.load_position` | `environment.load_position` | 0kg 빈 값은 `NONE`, 비영 하중 빈 값은 오류 |
| `ai1.condition.temperature_c` | `environment.temperature_c` | 현재 Model C/변화량 입력에는 미사용 |
| `ai1` 전체 | `ai1_result` | 원본 AI① 근거 보존, RiskScore 미합산 |
| `window_end_us` | `timestamp` | UTC ISO 8601로 변환 |
| `window_start*`, `window_end_us` | `event_time` | 분석 결과에 보존 |

`raw`는 XGBoost 입력도 아니고 현재 AI② 분석 캐시에 무제한 복제하지도 않는다. RAW의 Django 저장/전달 주체와 보존 기간은 미확정이다.

`schema_version`은 현재 존재 여부만 검사한다. 허용 버전 목록과 호환성 정책은 **CONFIRM_WITH_AI1**이다. timestamp의 timezone 유무 검증은 `realtime_processor.py`에서 수행한다.

## 기존 canonical 입력 필드

| 필드 | 자료형 | 필수 | 현재 허용/동작 | 누락 시 | 구현 |
|---|---|---:|---|---|---|
| `schema_version` | string | 예 | 값 제한 없음 | `FIELD_MISSING` | 구현됨 |
| `message_id` | string/null | 아니오 | 현재 출력에 보존, 중복 키 우선순위에는 사용하지 않음 | `bridge_type + window_id`로 중복 확인 | 부분 구현 |
| `bridge_type` | string | 예 | `BOX`, `TRUSS` | 입력 오류 | 구현됨 |
| `window_id` | non-empty string | 예 | 교량 내부 중복 키 | 실시간 처리 입력 오류 | 구현됨 |
| `timestamp` | timezone 포함 ISO 8601 string | 예 | 예: `2026-10-15T12:30:00+09:00` | 입력 오류 | 구현됨 |
| `features` | object | 예 | 아래 22개 Feature 모두 필요 | `FEATURE_MISSING` | 구현됨 |
| `environment` | object | 예 | 하중·위치 필수, 온도는 현재 변화량/Model C 입력에는 미사용 | 환경 입력 오류 | 부분 구현 |
| `ai1_result` | object/null | 아니오 | 원본값 보존만 수행 | `MISSING_OPTIONAL` | 구현됨 |

## `environment`

| 필드 | 자료형 | 단위/값 | 현재 동작 | 미확정 |
|---|---|---|---|---|
| `load_level_kg` | finite number | kg | 기준 조회 키 생성에 사용 | 센서 원값→단계화 규칙: CONFIRM_WITH_AI1 |
| `load_position` | string/null | `C`, `E`; 0kg 빈 값은 `NONE` | 0kg 빈 값은 `NONE`; 비영 하중 빈 값은 오류 | 물리 위치 코드 확장 여부: CONFIRM_WITH_AI1 |
| `temperature_c` | finite number 권장 | °C | 현재 입력 보존, 기준 조회·Model C Feature에는 미사용 | 실제 센서 보정·표본 시각: CONFIRM_WITH_AI1 |

## `features`: 정확한 순서

Model C artifact와 현재 `baseline_manager.FEATURES`의 순서는 다음과 정확히 일치해야 한다.

1. `acc01_rms_x`
2. `acc01_rms_y`
3. `acc01_rms_z`
4. `acc02_rms_x`
5. `acc02_rms_y`
6. `acc02_rms_z`
7. `acc01_p2p_x`
8. `acc01_p2p_y`
9. `acc01_p2p_z`
10. `acc02_p2p_x`
11. `acc02_p2p_y`
12. `acc02_p2p_z`
13. `acc01_dom_freq_x`
14. `acc01_dom_freq_y`
15. `acc01_dom_freq_z`
16. `acc02_dom_freq_x`
17. `acc02_dom_freq_y`
18. `acc02_dom_freq_z`
19. `acc01_damping_z`
20. `acc02_damping_z`
21. `cross_amp_ratio_L4_L2_z`
22. `cross_torsion_y_z_L2`

모든 값은 JSON number이며 NaN/Infinity는 허용하지 않는다. JSON object의 순서는 기술적으로 보장되지 않지만, AI②는 위 순서로 FeatureFrame을 재구성한다.

| Feature 그룹 | 정확한 이름 | 현재 확인된 단위 | 계산 방식 |
|---|---|---|---|
| RMS | `acc01_rms_x/y/z`, `acc02_rms_x/y/z` | m/s² | Window RMS라는 설명만 확인됨; detrending/필터/구간: CONFIRM_WITH_AI1 |
| P2P | `acc01_p2p_x/y/z`, `acc02_p2p_x/y/z` | m/s² | Window peak-to-peak라는 설명만 확인됨; peak 처리: CONFIRM_WITH_AI1 |
| 주 진동수 | `acc01_dom_freq_x/y/z`, `acc02_dom_freq_x/y/z` | Hz | dominant frequency 추출법·FFT 설정: CONFIRM_WITH_AI1 |
| 감쇠 | `acc01_damping_z`, `acc02_damping_z` | 무차원 감쇠비 | fitting 방법·유효 구간: CONFIRM_WITH_AI1 |
| 센서 간 응답 | `cross_amp_ratio_L4_L2_z`, `cross_torsion_y_z_L2` | 비율(무차원) | AI① 문서상 각각 L/4÷L/2 z 진폭비, L/2 y÷z 비; 분모 보호 규칙: CONFIRM_WITH_AI1 |

## `ai1_result`

현재 AI②는 이 객체를 검증·점수화하지 않고 보존한다.

| 필드 | 자료형 권장 | 확인된 의미 | 누락 동작 |
|---|---|---|---|
| `anomaly_score` | number | Isolation Forest 점수, 0~1, 클수록 이상 | null로 보존 |
| `decision_ratio` | number | `max(score/threshold, abs(z)/z_threshold)`, 1 초과 이상 | null로 보존 |
| `raw_prediction` | string | Window 단위 `Normal`/`Anomaly` | null로 보존 |
| `prediction` | string | 최근 3 Window 중 2개 이상 이상일 때 최종 `Anomaly` | null로 보존 |
| `model_version` | string | 현재 미확정 | CONFIRM_WITH_AI1 |
| `sensor_status` | object | 현재 미확정 | 그대로 보존, 해석하지 않음 |

`anomaly_score`와 `decision_ratio`는 현재 RiskScore에 합산되지 않는다.

## 실제 입력 예시

[BOX 합성 예시](../examples/ai1_to_ai2_box.json), [TRUSS 합성 예시](../examples/ai1_to_ai2_truss.json)를 참조한다. 두 파일은 AI①이 `POST /api/ai1/windows`로 그대로 보낼 수 있는 HTTP 요청 envelope이며, `tests/test_json_contract.py`가 `Ai1WindowRequest` 검증과 canonical 변환을 통과하는지 확인한다. Feature 값 `0.0`, AI① 점수 `0.0`, ID와 시각은 계약 형식 검증용 placeholder이며 실제 센서값이나 AI① 결과가 아니다. `raw.sensors`의 세부 필드는 API_SPEC 확인 전이므로 센서별 `null`로만 표시한다.

## 처리 정책

- HTTP ingress: 같은 `experiment_id + window_id`는 같은 `bridge_type`여야 한다. 다른 bridge면 `EXPERIMENT_WINDOW_BRIDGE_CONFLICT`다.
- 같은 `bridge_type + experiment_id + window_id`와 같은 내용: 캐시 결과 반환, `DUPLICATE_WINDOW`
- 같은 키와 다른 내용: `WINDOW_CONFLICT`; 재추론하지 않음
- legacy canonical 호출에 `experiment_id`가 없으면 기존 호환성을 위해 `bridge_type + window_id`를 키로 사용한다.
- 기준 조건 없음: `OUT_OF_CONDITION`; 다른 기준으로 대체하지 않음
- AI① 선택 필드 없음: Model C/변화량 처리는 계속 가능하며 `MISSING_OPTIONAL`
- `ai1.warming_up=true`: XGBoost/변화량 근거는 반환할 수 있으나 AI① `Normal`/`Anomaly`는 최종 상태로 해석하지 않는다. 출력 `ai1_status=PENDING_WARMUP`, data quality `AI1_WARMING_UP`을 추가한다.
