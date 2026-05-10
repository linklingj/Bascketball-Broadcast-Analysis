"""농구 경기 탐지 결과를 한 번에 후처리하는 통합 wrapper.

처리 순서:
1. 코트 바깥 객체 제거
2. 코트 안 객체만 이용해 공 후보 선택
3. 공 누락/가림/포물선 궤적 보정
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ball_tracker import correct_detections_with_ball_tracking
from court_filter import filter_detections_by_court


# 팀원용 역할 요약:
# 이 파일은 YOLO 결과 DataFrame에 두 가지 후처리를 순서대로 적용하는 연결 모듈입니다.
# 1) court_filter.py로 코트 밖 객체를 제거합니다.
# 2) ball_tracker.py로 공 후보를 하나의 안정적인 공 궤적으로 보정합니다.
# video_yolo.py/tracking_pipeline.py에서 직접 호출되며, 모델 학습에는 관여하지 않습니다.


def postprocess_basketball_detections(
    detections_df: pd.DataFrame,
    total_frames: Optional[int] = None,
    court_polygon: Optional[Sequence[Any]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    video_path: Optional[str] = "Video Project.mp4",
    polygon_is_normalized: bool = True,
    court_margin_px: float = 12.0,
    court_margin_by_class: Optional[Mapping[str, float]] = None,
    tracking_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """코트 필터와 공 추적 보정을 순서대로 적용한다."""

    # 먼저 코트 polygon 밖에 있는 선수/공/기타 객체를 제거한다. 이 단계는
    # 추론이 모두 끝난 뒤 DataFrame에서만 실행되므로 프레임별 YOLO 속도에 영향이 없다.
    # 1단계: 코트 polygon 밖에 있는 객체를 제거합니다.
    # 이 단계는 YOLO 추론이 끝난 뒤 DataFrame에서만 실행되므로 프레임별 추론 속도에 영향을 주지 않습니다.
    court_df = filter_detections_by_court(
        detections_df,
        court_polygon=court_polygon,
        frame_size=frame_size,
        video_path=video_path,
        polygon_is_normalized=polygon_is_normalized,
        margin_px=court_margin_px,
        margin_by_class=court_margin_by_class,
    )

    # 공 추적 보정의 기본 옵션이다. 필요하면 tracking_kwargs로 덮어쓸 수 있다.
    # 옵션 병합은 한 번만 수행해서 후처리 내부 반복 비용을 늘리지 않는다.
    # 2단계: 공 추적 보정 기본 옵션입니다. 필요하면 tracking_kwargs로 덮어쓸 수 있습니다.
    tracker_options = {
        "max_gap": 18,
        "auto_gravity": True,
        "enable_player_occlusion": True,
        "use_motion_association": True,
        "association_max_distance_px": 220.0,
        "association_gap_growth_px": 35.0,
        "player_center_radius_px": 90.0,
        "ball_classes": ("ball", "sports ball", "basketball", "frisbee"),
        "ball_label": "ball",
        "player_classes": ("player", "person"),
    }
    if tracking_kwargs:
        # 사용자 지정 옵션이 있으면 기본값보다 우선한다.
        tracker_options.update(dict(tracking_kwargs))

    # 코트 안 객체만 남은 DataFrame에서 공 row를 보정된 단일 궤적으로 교체한다.
    # 3단계: 코트 안 객체만 남긴 DataFrame에서 공 row를 보정된 단일 공 궤적으로 교체합니다.
    return correct_detections_with_ball_tracking(
        court_df,
        total_frames=total_frames,
        **tracker_options,
    )


def postprocess_detection_csv(
    input_csv: str = "video_detection.csv",
    output_csv: str = "video_detection_postprocessed.csv",
    total_frames: Optional[int] = None,
    court_polygon: Optional[Sequence[Any]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    video_path: Optional[str] = "Video Project.mp4",
    polygon_is_normalized: bool = True,
    court_margin_px: float = 12.0,
    court_margin_by_class: Optional[Mapping[str, float]] = None,
    tracking_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """CSV 파일을 읽어 통합 후처리를 실행하고 새 CSV로 저장한다."""
    # video_yolo.py가 만든 video_detection.csv를 그대로 입력으로 사용한다.
    detections_df = pd.read_csv(input_csv)
    result_df = postprocess_basketball_detections(
        detections_df,
        total_frames=total_frames,
        court_polygon=court_polygon,
        frame_size=frame_size,
        video_path=video_path,
        polygon_is_normalized=polygon_is_normalized,
        court_margin_px=court_margin_px,
        court_margin_by_class=court_margin_by_class,
        tracking_kwargs=tracking_kwargs,
    )
    # 후속 분석은 이 CSV를 사용하면 코트 밖 오검출과 공 누락 문제가 줄어든다.
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result_df


__all__ = [
    "postprocess_basketball_detections",
    "postprocess_detection_csv",
]


if __name__ == "__main__":
    # Run after video_yolo.py has created video_detection.csv.
    # Tune court_polygon for your broadcast camera if the default is too wide/narrow.
    postprocess_detection_csv()
