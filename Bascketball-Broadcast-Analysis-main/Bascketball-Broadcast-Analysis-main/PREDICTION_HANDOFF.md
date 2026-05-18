# 승률예측팀 전달 문서

## 목적

이 프로젝트는 농구 중계 영상에서 객체를 탐지하고, 팀별 이벤트 통계를 CSV로 저장합니다.
예측팀은 `event_summary.csv`를 주 입력 데이터로 사용해 승률예측 모델을 만들면 됩니다.

## 실행 방법

`video_yolo.py` 상단 설정값을 수정한 뒤 실행합니다.

```powershell
py video_yolo.py
```

주요 설정값:

```python
START_TIME_SECONDS = 1200
DURATION_SECONDS = 60
```

- `START_TIME_SECONDS = 0`, `DURATION_SECONDS = 0`: 전체 영상 분석
- `START_TIME_SECONDS = 1200`, `DURATION_SECONDS = 60`: 1200초부터 60초간 분석

## 출력 파일

실행 결과는 `OUTPUT_DIR`에 저장됩니다.

- `raw_detection_results.csv`: YOLO 객체탐지 원본 결과
- `improved_tracking_results.csv`: 후처리된 객체 결과와 팀 추정 정보
- `event_summary.csv`: 팀별 이벤트 집계 결과, 승률예측 주 입력
- `detection_bbox_video.mp4`: 탐지 박스가 표시된 검수용 영상

## 승률예측 입력으로 쓸 파일

예측 모델에는 `event_summary.csv`의 `row_type = team_summary` 행을 우선 사용합니다.
이 행은 팀별 통계가 한 줄에 모인 wide format이라 바로 feature로 쓰기 좋습니다.

주요 컬럼:

- `team_id`, `team_name`: 팀 구분값
- `shot_attempt_count`: 슛시도 수
- `shot_made_count`: 슛성공 수
- `shot_missed_count`: 슛실패 수
- `shot_success_rate_pct`: 슛성공률
- `pass_count`: 패스 수
- `steal_or_block_count`: 스틸/블락 수
- `foul_count`: 파울 수
- `rebound_count`: 리바운드 수
- `possession_time_sec`: 볼점유 시간
- `possession_pct`: 볼점유율

## row_type 의미

- `total_summary`: 전체 이벤트 종합
- `team_summary`: 팀별 이벤트 종합, 예측 모델 입력 추천
- `event_summary`: 이벤트 종류별 전체 count
- `team_event_summary`: 팀별, 이벤트 종류별 count
- `event`: 개별 이벤트 발생 기록
- `note`: 측정 방식과 주의사항

## 팀 구분 방식

현재 선수 탐지 모델은 팀을 별도 클래스로 구분하지 않습니다.
따라서 `improved_tracking_results.csv`에서 선수 유니폼 색상을 기준으로 `team_1`, `team_2`를 추정합니다.

- `team_1`: 더 밝은 유니폼 색상 군집
- `team_2`: 더 어두운 유니폼 색상 군집

팀명이 실제 홈/어웨이와 반드시 일치하는 것은 아니므로, 예측팀에서 실제 팀명과 매핑할 때 검수 영상과 함께 확인해야 합니다.

## 주의사항

이벤트는 객체 박스와 공 궤적을 기반으로 추정한 값입니다.
특히 파울은 휘슬, 오디오, 포즈 정보 없이 추정하므로 신뢰도가 낮습니다.

승률예측 모델을 만들려면 이 CSV의 feature 외에 실제 승패 label이 필요합니다.
여러 경기 영상에서 같은 방식으로 `event_summary.csv`를 모은 뒤, 각 경기의 실제 승패를 붙여 학습 데이터셋을 구성해야 합니다.

`point_prediction.ipynb`는 현재 영상 분석 실행에는 필요 없지만, 예측 모델 프로토타입 참고자료로 보관하는 것이 좋습니다.
