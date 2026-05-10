"""농구 경기 영상 객체 탐지 실행 파일.

이 파일은 사용자가 가장 먼저 실행하는 entrypoint입니다.
실제 YOLO 추론, 후처리, CSV/영상 저장 로직은 tracking_pipeline.py에 있고,
여기서는 명령행 옵션을 읽은 뒤 선택한 영상별로 파이프라인을 호출합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

from tracking_pipeline import run_yolo_tracking_pipeline


# 별도 옵션 없이 `py video_yolo.py`만 실행했을 때 사용할 기본 입력 영상입니다.
# 다른 영상을 분석하려면 `--input_video "영상파일.mp4"` 옵션을 사용하면 됩니다.
DEFAULT_VIDEO_PATH = "YTDown_YouTube_Media_nLKwrE_RGCc_002_720p.mp4"


def main() -> None:
    """명령행 옵션을 해석하고, 선택된 영상마다 탐지 파이프라인을 실행합니다."""

    args = parse_args()
    videos = list(resolve_input_videos(args))
    if not videos:
        raise FileNotFoundError("No input video files found.")

    for video_path in videos:
        output_paths = build_output_paths(video_path, args)
        print(f"Input video: {video_path}", flush=True)

        # 무거운 분석 로직은 tracking_pipeline.py가 담당합니다.
        # 이 파일은 팀원이 실행 옵션을 쉽게 바꿀 수 있게 하는 얇은 wrapper 역할만 합니다.
        run_yolo_tracking_pipeline(
            video_path=str(video_path),
            model_path=args.model_path,
            csv_output_path=output_paths["csv"],
            bbox_video_output_path=output_paths["bbox_video"],
            trajectory_video_output_path=output_paths["trajectory_video"],
            detection_summary_output_path=output_paths["summary"],
            tracker_config=args.tracker_config,
            conf_threshold=args.conf_threshold,
            imgsz=args.imgsz,
            ball_conf_threshold=args.ball_conf_threshold,
            ball_iou_threshold=args.ball_iou_threshold,
            ball_imgsz=args.ball_imgsz,
            enable_hoop_rescue=not args.disable_hoop_rescue,
            hoop_conf_threshold=args.hoop_conf_threshold,
            hoop_iou_threshold=args.hoop_iou_threshold,
            hoop_imgsz=args.hoop_imgsz,
            start_time_seconds=args.start_time_seconds,
            max_duration_seconds=args.max_duration_seconds,
            enable_rim_ball_rescue=not args.disable_rim_ball_rescue,
            rim_ball_rescue_conf_threshold=args.rim_ball_rescue_conf_threshold,
            rim_ball_rescue_iou_threshold=args.rim_ball_rescue_iou_threshold,
            rim_ball_rescue_margin_px=args.rim_ball_rescue_margin_px,
            rim_ball_rescue_imgsz=args.rim_ball_rescue_imgsz,
            enable_ball_tile_rescue=args.enable_ball_tile_rescue,
            ball_rescue_conf_threshold=args.ball_rescue_conf_threshold,
            ball_rescue_iou_threshold=args.ball_rescue_iou_threshold,
            ball_rescue_imgsz=args.ball_rescue_imgsz,
            device=args.device,
            use_half_if_cuda=not args.no_half,
            test_mode=args.test_mode,
            max_test_frames=args.max_test_frames,
        )


def parse_args() -> argparse.Namespace:
    """실행 옵션을 정의합니다.

    자주 사용하는 옵션:
    - --input_video: 분석할 영상 1개 지정
    - --process_all_videos: 현재 폴더의 영상 파일을 모두 분석
    - --output_dir: 결과 저장 폴더
    - --test_mode / --max_test_frames: 짧은 구간만 빠르게 테스트
    - --start_time_seconds: 인트로를 건너뛰고 경기 구간부터 분석
    - --device: GPU/CPU 직접 지정
    """

    parser = argparse.ArgumentParser(description="Run basketball YOLO detection and tracking.")
    parser.add_argument("--input_video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument(
        "--process_all_videos",
        action="store_true",
        help="Process every supported video file in the project folder.",
    )
    parser.add_argument("--model_path", default="player_detector.pt", help="YOLO model path.")
    parser.add_argument("--output_dir", default="runs/detect", help="Output directory.")
    parser.add_argument(
        "--tracker_config",
        default="bytetrack.yaml",
        help="Ultralytics tracker config, for example bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.12,
        help="Confidence threshold for player/referee/hoop detection.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO image size for non-ball objects.")
    parser.add_argument(
        "--ball_conf_threshold",
        type=float,
        default=0.02,
        help="Low confidence threshold for the dedicated ball detection pass.",
    )
    parser.add_argument(
        "--ball_iou_threshold",
        type=float,
        default=0.65,
        help="NMS IoU threshold for the dedicated ball detection pass.",
    )
    parser.add_argument("--ball_imgsz", type=int, default=1536, help="YOLO image size for ball detection.")
    parser.add_argument(
        "--disable_hoop_rescue",
        action="store_true",
        help="Disable the low-confidence hoop rescue pass.",
    )
    parser.add_argument(
        "--hoop_conf_threshold",
        type=float,
        default=0.035,
        help="Low confidence threshold for the dedicated hoop rescue pass.",
    )
    parser.add_argument(
        "--hoop_iou_threshold",
        type=float,
        default=0.65,
        help="NMS IoU threshold for the dedicated hoop rescue pass.",
    )
    parser.add_argument(
        "--hoop_imgsz",
        type=int,
        default=1280,
        help="YOLO image size for the dedicated hoop rescue pass.",
    )
    parser.add_argument(
        "--start_time_seconds",
        type=float,
        default=0.0,
        help="Start processing after this many seconds.",
    )
    parser.add_argument(
        "--max_duration_seconds",
        type=float,
        default=None,
        help="Maximum seconds to process. Omit for the full video.",
    )
    parser.add_argument(
        "--enable_ball_tile_rescue",
        action="store_true",
        help="Enable crop-based rescue detection for missed ball frames.",
    )
    parser.add_argument(
        "--disable_rim_ball_rescue",
        action="store_true",
        help="Disable low-threshold ball rescue around detected rims.",
    )
    parser.add_argument(
        "--rim_ball_rescue_conf_threshold",
        type=float,
        default=0.006,
        help="Confidence threshold used only inside rim rescue crops.",
    )
    parser.add_argument(
        "--rim_ball_rescue_iou_threshold",
        type=float,
        default=0.70,
        help="NMS IoU threshold used only inside rim rescue crops.",
    )
    parser.add_argument(
        "--rim_ball_rescue_margin_px",
        type=float,
        default=220.0,
        help="Pixel margin around a hoop bbox for rim rescue detection.",
    )
    parser.add_argument(
        "--rim_ball_rescue_imgsz",
        type=int,
        default=1280,
        help="YOLO image size for rim rescue crops.",
    )
    parser.add_argument(
        "--ball_rescue_conf_threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for optional ball rescue crops.",
    )
    parser.add_argument(
        "--ball_rescue_iou_threshold",
        type=float,
        default=0.70,
        help="NMS IoU threshold for optional ball rescue crops.",
    )
    parser.add_argument(
        "--ball_rescue_imgsz",
        type=int,
        default=1280,
        help="YOLO image size for optional ball rescue crops.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example cpu, 0, cuda, or cuda:0.",
    )
    parser.add_argument("--no_half", action="store_true", help="Disable half precision on CUDA.")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Process only the first --max_test_frames frames.",
    )
    parser.add_argument(
        "--max_test_frames",
        type=int,
        default=300,
        help="Frame limit used with --test_mode.",
    )
    parser.add_argument(
        "--save_detection_summary",
        action="store_true",
        help="Also save a separate detection_summary.csv file.",
    )
    return parser.parse_args()


def resolve_input_videos(args: argparse.Namespace) -> Iterable[Path]:
    """분석할 입력 영상 목록을 반환합니다."""

    if args.process_all_videos:
        patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        videos = []
        for pattern in patterns:
            videos.extend(Path(".").glob(pattern))
        return sorted(videos)
    return [Path(args.input_video)]


def build_output_paths(video_path: Path, args: argparse.Namespace) -> dict[str, str | None]:
    """입력 영상명과 옵션을 기준으로 결과 파일 경로를 만듭니다."""

    output_dir = Path(args.output_dir)
    stem = safe_stem(video_path.stem)

    # 여러 영상을 한 번에 분석할 때는 결과물이 서로 덮어쓰이지 않도록 영상명 prefix를 붙입니다.
    # 단일 영상 실행에서는 기존처럼 tracking_results.csv 같은 기본 이름을 유지합니다.
    prefix = f"{stem}_" if args.process_all_videos else ""

    return {
        "csv": str(output_dir / f"{prefix}tracking_results.csv"),
        "bbox_video": str(output_dir / f"{prefix}tracking_visualization.mp4"),
        "trajectory_video": str(output_dir / f"{prefix}ball_trajectory_visualization.mp4"),
        "summary": str(output_dir / f"{prefix}detection_summary.csv")
        if args.save_detection_summary
        else None,
    }


def safe_stem(stem: str) -> str:
    """파일명을 결과 prefix로 안전하게 사용할 수 있도록 정리합니다."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or "video"


if __name__ == "__main__":
    main()
