# -*- coding: utf-8 -*-
"""농구 중계 영상 분석 실행 파일.

아래 설정값만 바꾼 뒤 실행합니다:
    py video_yolo.py

START_TIME_SECONDS = 0, DURATION_SECONDS = 0이면 전체 영상을 처리합니다.
START_TIME_SECONDS = 1200, DURATION_SECONDS = 60이면 1200초부터 60초간 처리합니다.
"""

from __future__ import annotations

from pathlib import Path

from tracking_pipeline import run_yolo_tracking_pipeline


# 실행 전에 여기 값만 수정하면 됩니다.
INPUT_VIDEO_PATH = "YTDown_YouTube_Kyrie-Irving-ERUPTS-For-A-CAREER-HIGH-60_Media_4Pc01w1n9Mg_002_720p.mp4"
OUTPUT_DIR = "runs/verify_60s"

START_TIME_SECONDS = 1200
DURATION_SECONDS = 60

# 사용하는 모델 파일입니다.
PLAYER_MODEL_PATH = "player_detector.pt"
BALL_MODEL_PATH = "best.pt"
SECONDARY_BALL_MODEL_PATH = "ball_detector_model.pt"
COURT_MODEL_PATH = "court_keypoint_detector.pt"


def main() -> None:
    """설정된 영상 1개를 분석합니다."""

    validate_runtime_settings()

    output_paths = build_output_paths(Path(OUTPUT_DIR))
    duration_seconds = resolve_duration_seconds()

    duration_text = "전체 영상" if duration_seconds is None else f"{duration_seconds:g}초"
    print(f"입력 영상: {INPUT_VIDEO_PATH}", flush=True)
    print(
        f"{float(START_TIME_SECONDS):g}초부터 {duration_text} 처리합니다.",
        flush=True,
    )

    run_yolo_tracking_pipeline(
        video_path=INPUT_VIDEO_PATH,
        model_path=PLAYER_MODEL_PATH,
        ball_model_path=BALL_MODEL_PATH,
        secondary_ball_model_path=SECONDARY_BALL_MODEL_PATH,
        court_model_path=COURT_MODEL_PATH,
        raw_csv_output_path=output_paths["raw_csv"],
        csv_output_path=output_paths["improved_csv"],
        event_csv_output_path=output_paths["event_csv"],
        bbox_video_output_path=output_paths["bbox_video"],
        trajectory_video_output_path=None,
        detection_summary_output_path=None,
        start_time_seconds=float(START_TIME_SECONDS),
        max_duration_seconds=duration_seconds,
    )


def validate_runtime_settings() -> None:
    """실행 전 사용자가 수정한 설정값이 올바른지 확인합니다."""

    if not Path(INPUT_VIDEO_PATH).exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {INPUT_VIDEO_PATH}")

    if float(START_TIME_SECONDS) < 0:
        raise ValueError("START_TIME_SECONDS는 0 이상이어야 합니다.")

    if float(DURATION_SECONDS) < 0:
        raise ValueError("DURATION_SECONDS는 0 이상이어야 합니다.")


def resolve_duration_seconds() -> float | None:
    """DURATION_SECONDS가 0이면 영상 끝까지 처리하도록 None을 반환합니다."""

    duration_seconds = float(DURATION_SECONDS)
    if duration_seconds == 0:
        return None
    return duration_seconds


def build_output_paths(output_dir: Path) -> dict[str, str]:
    """사용자가 확인할 4개 결과 파일 경로를 만듭니다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw_csv": str(output_dir / "raw_detection_results.csv"),
        "improved_csv": str(output_dir / "improved_tracking_results.csv"),
        "event_csv": str(output_dir / "event_summary.csv"),
        "bbox_video": str(output_dir / "detection_bbox_video.mp4"),
    }


if __name__ == "__main__":
    main()
