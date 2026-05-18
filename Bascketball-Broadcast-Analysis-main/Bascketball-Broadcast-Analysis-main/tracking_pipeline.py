"""YOLO basketball detection pipeline.

This module keeps video_yolo.py small and owns the real work:
- Track players, referees, and hoops with ByteTrack.
- Detect the ball with a separate low-confidence/high-resolution pass.
- Save a cleaned CSV.
- Save one bbox visualization video.
- Save one ball trajectory visualization video.

성능을 유지하기 위해 프레임 루프 안에서는 모델 호출, 최소한의 후보 필터링,
리스트 append만 수행한다. DataFrame 정리와 CSV 저장은 영상 처리가 끝난 뒤 한 번에 한다.
"""

# 팀원용 빠른 흐름 요약:
# video_yolo.py -> run_yolo_tracking_pipeline() -> YOLO 추론 -> 후처리 -> CSV/영상 저장
# 이 파일에서 가장 중요한 함수는 run_yolo_tracking_pipeline()입니다.
# 공 검출은 detect_ball_records(), 공 후보 정리는 filter_ball_candidates(),
# 최종 출력 저장은 make_output_dataframe()/render_bbox_video()/render_ball_trajectory_video()가 담당합니다.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import pandas as pd
from ultralytics import YOLO

from ball_coordinate_refiner import BallCoordinateRefiner, build_adaptive_refiner_config
from basketball_event_analyzer import (
    assign_fallback_player_track_ids,
    assign_player_teams_from_video,
    write_event_measurements,
)
from basketball_postprocess import postprocess_basketball_detections
from court_filter import CourtObjectFilter, DEFAULT_BROADCAST_COURT_POLYGON_NORM


CSV_COLUMNS = [
    "frame",
    "track_id",
    "class",
    "confidence",
    "x_center",
    "y_center",
    "x1",
    "y1",
    "x2",
    "y2",
]

# Model class names are normalized into these project-level class names.
CLASS_ALIASES = {
    "ball": "ball",
    "sports ball": "ball",
    "basketball": "ball",
    "frisbee": "ball",
    "player": "player",
    "person": "player",
    "ref": "referee",
    "referee": "referee",
    "hoop": "hoop",
    "rim": "hoop",
    "basket": "hoop",
    "basketball hoop": "hoop",
}

TARGET_CLASSES = {"ball", "player", "referee", "hoop"}
NON_BALL_CLASSES = {"player", "referee", "hoop"}
BALL_CLASSES = {"ball"}

# OpenCV colors are BGR.
CLASS_COLORS = {
    "ball": (0, 80, 255),
    "player": (255, 120, 40),
    "referee": (80, 255, 255),
    "hoop": (80, 255, 80),
}

TEAM_COLORS = {
    "1": (255, 80, 80),
    "2": (80, 170, 255),
}

FULL_FRAME_POLYGON_NORM = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
]


def run_yolo_tracking_pipeline(
    video_path: str = "Video Project.mp4",
    model_path: str = "player_detector.pt",
    ball_model_path: Optional[str] = "best.pt",
    secondary_ball_model_path: Optional[str] = "ball_detector_model.pt",
    court_model_path: Optional[str] = "court_keypoint_detector.pt",
    csv_output_path: str = "runs/detect/improved_tracking_results.csv",
    raw_csv_output_path: Optional[str] = "runs/detect/raw_detection_results.csv",
    event_csv_output_path: Optional[str] = "runs/detect/event_measurements.csv",
    bbox_video_output_path: str = "runs/detect/tracking_visualization.mp4",
    trajectory_video_output_path: Optional[str] = None,
    detection_summary_output_path: Optional[str] = None,
    tracker_config: str = "bytetrack.yaml",
    conf_threshold: float = 0.12,
    imgsz: int = 1280,
    ball_conf_threshold: float = 0.02,
    ball_iou_threshold: float = 0.65,
    ball_imgsz: int = 1536,
    max_det: int = 300,
    ball_max_det: int = 120,
    enable_hoop_rescue: bool = True,
    hoop_conf_threshold: float = 0.035,
    hoop_iou_threshold: float = 0.65,
    hoop_imgsz: int = 1280,
    hoop_max_det: int = 40,
    enable_player_rescue: bool = True,
    player_rescue_conf_threshold: float = 0.045,
    player_rescue_iou_threshold: float = 0.65,
    player_rescue_imgsz: int = 1280,
    player_rescue_max_det: int = 160,
    player_rescue_min_players: int = 8,
    device: Optional[str] = None,
    use_half_if_cuda: bool = True,
    start_time_seconds: float = 0.0,
    max_duration_seconds: Optional[float] = None,
    enable_rim_ball_rescue: bool = True,
    rim_ball_rescue_conf_threshold: float = 0.006,
    rim_ball_rescue_iou_threshold: float = 0.70,
    rim_ball_rescue_margin_px: float = 220.0,
    rim_ball_rescue_imgsz: int = 1280,
    enable_ball_tile_rescue: bool = False,
    ball_rescue_conf_threshold: float = 0.01,
    ball_rescue_iou_threshold: float = 0.70,
    ball_rescue_imgsz: int = 1280,
    court_conf_threshold: float = 0.15,
    court_keypoint_conf_threshold: float = 0.05,
    court_imgsz: int = 640,
    court_update_interval_frames: int = 10,
    court_polygon_expand_px: float = 35.0,
    test_mode: bool = False,
    max_test_frames: int = 300,
) -> Dict[str, Optional[str]]:
    """영상 1개를 끝까지 분석하고 결과 파일을 저장합니다.

    반환값은 생성된 결과 파일 경로입니다. 팀원이 분석 흐름을 볼 때는 이 함수 안의
    큰 주석 단위(입력 검증 -> YOLO 추론 -> 후처리 -> 저장)를 따라가면 됩니다.
    """

    # 1) 입력 파일과 모델 파일이 실제로 존재하는지 먼저 확인합니다.
    _validate_input_file(video_path, "input video")
    _validate_input_file(model_path, "YOLO model")
    if ball_model_path:
        _validate_input_file(ball_model_path, "ball YOLO model")
    if secondary_ball_model_path:
        _validate_input_file(secondary_ball_model_path, "secondary ball YOLO model")
    if court_model_path:
        _validate_input_file(court_model_path, "court keypoint model")

    # 2) 원본 영상의 fps/해상도/프레임 수를 읽어 결과 영상 저장 설정에 그대로 사용합니다.
    metadata = read_video_metadata(video_path)
    frame_size = (metadata["width"], metadata["height"])
    adaptive_profile = build_adaptive_processing_profile(metadata)

    # 영상별 고정 튜닝을 하지 않기 위해 해상도/FPS 기반 profile을 만든다.
    # CLI로 사용자가 더 강한 설정을 준 경우는 존중하고, 기본값보다 부족한 부분만 자동 보강한다.
    effective_conf_threshold = min(float(conf_threshold), float(adaptive_profile["non_ball_conf_threshold"]))
    effective_imgsz = max(int(imgsz), int(adaptive_profile["imgsz"]))
    effective_ball_imgsz = max(int(ball_imgsz), int(adaptive_profile["ball_imgsz"]))
    effective_hoop_imgsz = max(int(hoop_imgsz), int(adaptive_profile["hoop_imgsz"]))
    effective_hoop_conf_threshold = min(
        float(hoop_conf_threshold),
        float(adaptive_profile["hoop_conf_threshold"]),
    )
    effective_player_rescue_imgsz = max(
        int(player_rescue_imgsz),
        int(adaptive_profile["imgsz"]),
    )
    effective_player_rescue_conf_threshold = min(
        float(player_rescue_conf_threshold),
        float(adaptive_profile["player_rescue_conf_threshold"]),
    )
    effective_max_det = max(int(max_det), int(adaptive_profile["max_det"]))
    effective_rim_rescue_margin_px = max(
        float(rim_ball_rescue_margin_px),
        float(adaptive_profile["rim_rescue_margin_px"]),
    )
    effective_rim_rescue_imgsz = max(
        int(rim_ball_rescue_imgsz),
        int(adaptive_profile["rim_rescue_imgsz"]),
    )
    effective_ball_rescue_imgsz = max(
        int(ball_rescue_imgsz),
        int(adaptive_profile["ball_rescue_imgsz"]),
    )
    effective_enable_ball_tile_rescue = bool(enable_ball_tile_rescue or adaptive_profile["enable_tile_rescue"])

    # 3) YOLO 모델을 로드하고, CUDA GPU가 가능하면 자동으로 GPU를 사용합니다.
    model = YOLO(model_path)
    ball_model = YOLO(ball_model_path) if ball_model_path else model
    secondary_ball_model = (
        YOLO(secondary_ball_model_path)
        if should_use_secondary_model(ball_model_path, secondary_ball_model_path)
        else None
    )
    court_model = YOLO(court_model_path) if court_model_path else None
    resolved_device = resolve_yolo_device(device)
    use_half = use_half_if_cuda and resolved_device != "cpu"
    print(f"YOLO device: {resolved_device}, half precision: {use_half}", flush=True)
    if ball_model_path:
        print(f"Ball model: {ball_model_path}", flush=True)
    if secondary_ball_model_path and secondary_ball_model is not None:
        print(f"Secondary ball model: {secondary_ball_model_path}", flush=True)
    if court_model_path:
        print(f"Court keypoint model: {court_model_path}", flush=True)

    # 모델이 가진 클래스 이름을 한 번만 해석한다. 프레임마다 문자열 매칭을 반복하지
    # 않도록 클래스 id 목록을 미리 만들어 추론 호출에 바로 넘긴다.
    # 4) 모델 클래스 이름을 프로젝트 표준 클래스명(ball/player/referee/hoop)으로 매핑합니다.
    #    프레임마다 문자열 매칭을 반복하지 않도록 class id 목록을 미리 계산합니다.
    non_ball_class_ids = resolve_class_ids(model.names, NON_BALL_CLASSES) or None
    player_class_ids = resolve_class_ids(model.names, {"player"})
    ball_class_ids = resolve_class_ids(ball_model.names, BALL_CLASSES)
    secondary_ball_class_ids = (
        resolve_class_ids(secondary_ball_model.names, BALL_CLASSES)
        if secondary_ball_model is not None
        else []
    )
    hoop_class_ids = resolve_class_ids(model.names, {"hoop"})
    if not ball_class_ids:
        raise ValueError(f"No ball class found in ball model: {ball_model_path or model_path}")
    if secondary_ball_model is not None and not secondary_ball_class_ids:
        raise ValueError(f"No ball class found in secondary ball model: {secondary_ball_model_path}")
    print(f"Detector classes: {dict(model.names)}", flush=True)
    if has_team_specific_classes(model.names):
        print("Team-specific detector classes found; normalized output still uses project player labels.", flush=True)
    else:
        print("Team-specific detector classes not found; assigning teams from player uniform colors.", flush=True)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    start_frame = max(0, int(round(float(start_time_seconds) * float(metadata["fps"]))))
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    max_processing_frames = resolve_processing_frame_limit(
        fps=float(metadata["fps"]),
        max_duration_seconds=max_duration_seconds,
        test_mode=test_mode,
        max_test_frames=max_test_frames,
    )

    # 5) 프레임별 추론 결과는 dict 리스트로 누적하고, 마지막에 DataFrame으로 변환합니다.
    records: List[Dict[str, Any]] = []
    frame_id = 0
    current_court_polygon: Optional[List[Tuple[float, float]]] = None
    court_margin_by_class = {
        "ball": float(adaptive_profile["court_ball_margin_px"]),
        "sports ball": float(adaptive_profile["court_ball_margin_px"]),
        "basketball": float(adaptive_profile["court_ball_margin_px"]),
        "frisbee": float(adaptive_profile["court_ball_margin_px"]),
        "hoop": float(adaptive_profile["court_hoop_margin_px"]),
        "rim": float(adaptive_profile["court_hoop_margin_px"]),
        "basket": float(adaptive_profile["court_hoop_margin_px"]),
        "basketball hoop": float(adaptive_profile["court_hoop_margin_px"]),
    }

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Players/referees/hoops need stable IDs, so they use ByteTrack.
        tracked_results = model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            conf=effective_conf_threshold,
            imgsz=effective_imgsz,
            classes=non_ball_class_ids,
            max_det=effective_max_det,
            device=resolved_device,
            half=use_half,
            verbose=False,
        )
        frame_records = extract_detection_records(tracked_results, model.names, frame_id)

        if enable_player_rescue and count_class_records(frame_records, "player") < int(player_rescue_min_players):
            player_records = detect_player_records(
                frame=frame,
                model=model,
                model_names=model.names,
                frame_id=frame_id,
                frame_size=frame_size,
                player_class_ids=player_class_ids,
                device=resolved_device,
                use_half=use_half,
                confidence_threshold=effective_player_rescue_conf_threshold,
                iou_threshold=player_rescue_iou_threshold,
                imgsz=effective_player_rescue_imgsz,
                max_det=player_rescue_max_det,
            )
            frame_records = merge_player_records(frame_records, player_records, frame_size)

        # 링은 화면 상단에 작게 보이거나 백보드/관중석과 섞여 일반 track 단계에서 빠질 수 있다.
        # 해당 프레임에 링이 없을 때만 낮은 confidence의 링 전용 pass를 추가로 돌려 공 rescue와 슛 분석 context를 보강한다.
        if enable_hoop_rescue and not has_class_record(frame_records, "hoop"):
            hoop_records = detect_hoop_records(
                frame=frame,
                model=model,
                model_names=model.names,
                frame_id=frame_id,
                frame_size=frame_size,
                hoop_class_ids=hoop_class_ids,
                device=resolved_device,
                use_half=use_half,
                confidence_threshold=effective_hoop_conf_threshold,
                iou_threshold=hoop_iou_threshold,
                imgsz=effective_hoop_imgsz,
                max_det=hoop_max_det,
            )
            frame_records = merge_hoop_records(frame_records, hoop_records, frame_size)

        # The ball is small and often low-confidence, so run a dedicated ball-only pass.
        # This does not assign a stable raw ID; ball_tracker.py later turns it into track_id 0.
        ball_records = detect_ball_records(
            frame=frame,
            model=ball_model,
            model_names=ball_model.names,
            frame_id=frame_id,
            frame_size=frame_size,
            ball_class_ids=ball_class_ids,
            context_records=frame_records,
            device=resolved_device,
            use_half=use_half,
            ball_conf_threshold=ball_conf_threshold,
            ball_iou_threshold=ball_iou_threshold,
            ball_imgsz=effective_ball_imgsz,
            ball_max_det=ball_max_det,
            enable_rim_ball_rescue=enable_rim_ball_rescue,
            rim_rescue_conf_threshold=rim_ball_rescue_conf_threshold,
            rim_rescue_iou_threshold=rim_ball_rescue_iou_threshold,
            rim_rescue_margin_px=effective_rim_rescue_margin_px,
            rim_rescue_imgsz=effective_rim_rescue_imgsz,
            enable_tile_rescue=effective_enable_ball_tile_rescue,
            rescue_conf_threshold=ball_rescue_conf_threshold,
            rescue_iou_threshold=ball_rescue_iou_threshold,
            rescue_imgsz=effective_ball_rescue_imgsz,
            model_label=Path(ball_model_path or model_path).name,
            model_priority_bonus=0.0,
        )
        if secondary_ball_model is not None:
            secondary_ball_records = detect_ball_records(
                frame=frame,
                model=secondary_ball_model,
                model_names=secondary_ball_model.names,
                frame_id=frame_id,
                frame_size=frame_size,
                ball_class_ids=secondary_ball_class_ids,
                context_records=frame_records,
                device=resolved_device,
                use_half=use_half,
                ball_conf_threshold=ball_conf_threshold,
                ball_iou_threshold=ball_iou_threshold,
                ball_imgsz=effective_ball_imgsz,
                ball_max_det=ball_max_det,
                enable_rim_ball_rescue=enable_rim_ball_rescue,
                rim_rescue_conf_threshold=rim_ball_rescue_conf_threshold,
                rim_rescue_iou_threshold=rim_ball_rescue_iou_threshold,
                rim_rescue_margin_px=effective_rim_rescue_margin_px,
                rim_rescue_imgsz=effective_rim_rescue_imgsz,
                enable_tile_rescue=effective_enable_ball_tile_rescue,
                rescue_conf_threshold=ball_rescue_conf_threshold,
                rescue_iou_threshold=ball_rescue_iou_threshold,
                rescue_imgsz=effective_ball_rescue_imgsz,
                model_label=Path(secondary_ball_model_path).name,
                model_priority_bonus=0.0,
            )
            ball_records = filter_ball_candidates(
                [*ball_records, *secondary_ball_records],
                frame_size,
                context_records=frame_records,
            )

        combined_records = frame_records + ball_records
        if court_model is not None:
            if should_update_court_polygon(
                frame_id=frame_id,
                interval_frames=court_update_interval_frames,
                current_polygon=current_court_polygon,
            ):
                detected_polygon = detect_court_polygon(
                    frame=frame,
                    model=court_model,
                    frame_size=frame_size,
                    device=resolved_device,
                    use_half=use_half,
                    confidence_threshold=court_conf_threshold,
                    keypoint_conf_threshold=court_keypoint_conf_threshold,
                    imgsz=court_imgsz,
                    expand_px=court_polygon_expand_px,
                )
                if detected_polygon is not None:
                    current_court_polygon = detected_polygon

            combined_records = filter_records_by_court_polygon(
                combined_records,
                frame_size=frame_size,
                court_polygon=current_court_polygon,
                margin_by_class=court_margin_by_class,
            )
        records.extend(combined_records)

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"YOLO processing: {frame_id} frames", flush=True)

        if max_processing_frames is not None and frame_id >= max_processing_frames:
            break

    capture.release()

    # 6) YOLO 원본 결과를 표준 CSV 컬럼 구조로 정리합니다.
    raw_df = make_tracking_dataframe(records)
    if raw_csv_output_path:
        Path(raw_csv_output_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_csv_output_path, index=False, encoding="utf-8-sig")

    print("Postprocessing: court filter + ball tracking correction", flush=True)
    # 7) court_filter.py와 ball_tracker.py를 연결해 코트 밖 객체 제거와 공 궤적 보정을 수행합니다.
    postprocessed_df = postprocess_basketball_detections(
        raw_df,
        total_frames=frame_id,
        frame_size=frame_size,
        video_path=video_path,
        court_polygon=FULL_FRAME_POLYGON_NORM if court_model is not None else None,
        court_margin_by_class=court_margin_by_class,
        tracking_kwargs={
            "ball_classes": ("ball", "sports ball", "basketball", "frisbee"),
            "ball_label": "ball",
            "player_classes": ("player", "person"),
            "enable_player_occlusion": True,
            "enable_rim_proximity_boost": True,
            "rim_classes": ("hoop", "rim", "basketball hoop"),
            "rim_proximity_px": float(adaptive_profile["rim_proximity_px"]),
            "rim_interpolation_extra_gap": int(adaptive_profile["rim_interpolation_extra_gap"]),
            "use_motion_association": True,
            "association_max_distance_px": float(adaptive_profile["association_max_distance_px"]),
            "association_gap_growth_px": float(adaptive_profile["association_gap_growth_px"]),
            "association_confidence_weight": 25.0,
            "association_restart_after_frames": int(adaptive_profile["association_restart_after_frames"]),
            "association_restart_min_confidence": 0.006,
            "max_gap": int(adaptive_profile["max_gap"]),
            "max_prediction_frames": int(adaptive_profile["max_prediction_frames"]),
            "auto_gravity": True,
        },
    )
    postprocessed_df = stabilize_hoop_detections(
        postprocessed_df,
        total_frames=frame_id,
        frame_size=frame_size,
        fps=float(metadata["fps"]),
    )

    # 8) 최종 CSV에 필요한 컬럼만 남기고, 보정된 공 좌표에 bbox가 없으면 작은 bbox를 생성합니다.
    final_df = make_output_dataframe(postprocessed_df)
    final_df = remove_invalid_ball_positions(final_df, frame_size)
    final_df = remove_rim_false_ball_positions(final_df, frame_size, fps=float(metadata["fps"]))
    final_df = assign_fallback_player_track_ids(final_df, frame_size)
    print("Refining ball coordinates for clean-data output", flush=True)
    # 통계 계산 전 단계에서 사용할 깨끗한 공 좌표를 별도 파일로 생성합니다.
    # raw_df는 원본 공 후보 검수용 CSV에 사용하고, final_df는 이상치 제거/보간/smoothing의 입력으로 사용합니다.
    refiner = BallCoordinateRefiner(build_adaptive_refiner_config(frame_size, float(metadata["fps"])))
    improved_df = refiner.create_cleaned_tracking_results(
        tracking_df=final_df,
        frame_size=frame_size,
        total_frames=frame_id,
        output_path=csv_output_path,
    )
    improved_df = remove_invalid_ball_positions(improved_df, frame_size)
    improved_df = remove_rim_false_ball_positions(improved_df, frame_size, fps=float(metadata["fps"]))

    print("Assigning player teams from uniform colors", flush=True)
    improved_df, team_summary = assign_player_teams_from_video(
        tracking_df=improved_df,
        video_path=video_path,
        frame_size=frame_size,
        start_frame=start_frame,
    )
    Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
    improved_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print_team_assignment_summary(team_summary)

    summary_df = make_detection_summary(
        improved_df,
        processed_frames=frame_id,
        fps=float(metadata["fps"]),
        video_path=video_path,
    )
    save_detection_summary(summary_df, detection_summary_output_path)
    print_detection_summary(summary_df)

    event_df: Optional[pd.DataFrame] = None
    if event_csv_output_path:
        print("Measuring basketball events from cleaned tracking results", flush=True)
        Path(event_csv_output_path).parent.mkdir(parents=True, exist_ok=True)
        event_df = write_event_measurements(
            tracking_df=improved_df,
            output_csv_path=event_csv_output_path,
            fps=float(metadata["fps"]),
            frame_size=frame_size,
            video_path=video_path,
        )
        print_event_measurement_summary(event_df)

    # 9) 팀원이 눈으로 검수할 수 있도록 박스 영상과 공 궤적 영상을 생성합니다.
    print("Rendering bbox video from cleaned tracking results", flush=True)
    render_bbox_video(
        video_path=video_path,
        output_path=bbox_video_output_path,
        tracking_df=improved_df,
        fps=metadata["fps"],
        frame_size=frame_size,
        max_frames=frame_id,
        start_frame=start_frame,
    )

    if trajectory_video_output_path:
        print("Rendering ball trajectory video from cleaned tracking results", flush=True)
        render_ball_trajectory_video(
            video_path=video_path,
            output_path=trajectory_video_output_path,
            tracking_df=improved_df,
            fps=metadata["fps"],
            frame_size=frame_size,
            max_frames=frame_id,
            start_frame=start_frame,
        )

    if raw_csv_output_path:
        print(f"Raw detection CSV saved: {raw_csv_output_path}")
    print(f"Improved tracking CSV saved: {csv_output_path}")
    if event_csv_output_path:
        print(f"Event measurement CSV saved: {event_csv_output_path}")
    print(f"BBox visualization saved: {bbox_video_output_path}")
    if trajectory_video_output_path:
        print(f"Ball trajectory visualization saved: {trajectory_video_output_path}")
    if detection_summary_output_path:
        print(f"Detection summary saved: {detection_summary_output_path}")

    return {
        "raw_csv": raw_csv_output_path,
        "improved_csv": csv_output_path,
        "event_csv": event_csv_output_path,
        "bbox_video": bbox_video_output_path,
        "trajectory_video": trajectory_video_output_path,
        "detection_summary": detection_summary_output_path,
    }


def make_detection_summary(
    df: pd.DataFrame,
    processed_frames: int,
    fps: float,
    video_path: str,
) -> pd.DataFrame:
    """Build per-class counts and frame coverage for checking full-video detection."""

    if df.empty or "class" not in df.columns:
        return pd.DataFrame(
            columns=[
                "video_path",
                "class",
                "count",
                "detected_frames",
                "processed_frames",
                "frame_coverage_pct",
                "first_frame",
                "last_frame",
                "first_time_sec",
                "last_time_sec",
                "positive_confidence_rows",
                "mean_confidence",
                "median_confidence",
                "min_confidence",
                "max_confidence",
                "unique_track_ids",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for class_name, class_df in df.groupby("class", sort=True):
        frames = pd.to_numeric(class_df["frame"], errors="coerce").dropna()
        confidence = pd.to_numeric(
            class_df.get("confidence", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        track_ids = pd.to_numeric(
            class_df.get("track_id", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        detected_frames = int(frames.nunique())
        first_frame = int(frames.min()) if not frames.empty else None
        last_frame = int(frames.max()) if not frames.empty else None
        rows.append(
            {
                "video_path": video_path,
                "class": class_name,
                "count": int(len(class_df)),
                "detected_frames": detected_frames,
                "processed_frames": int(processed_frames),
                "frame_coverage_pct": round((detected_frames / processed_frames) * 100.0, 2)
                if processed_frames > 0
                else 0.0,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "first_time_sec": round(first_frame / fps, 3)
                if first_frame is not None and fps > 0
                else None,
                "last_time_sec": round(last_frame / fps, 3)
                if last_frame is not None and fps > 0
                else None,
                "positive_confidence_rows": int(confidence.gt(0).sum()),
                "mean_confidence": round(float(confidence.mean()), 4) if len(confidence) else 0.0,
                "median_confidence": round(float(confidence.median()), 4) if len(confidence) else 0.0,
                "min_confidence": round(float(confidence.min()), 4) if len(confidence) else 0.0,
                "max_confidence": round(float(confidence.max()), 4) if len(confidence) else 0.0,
                "unique_track_ids": int(track_ids.nunique()) if not track_ids.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def save_detection_summary(summary_df: pd.DataFrame, output_path: Optional[str]) -> None:
    """Save the detection summary CSV when requested."""

    if not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output, index=False, encoding="utf-8-sig")


def print_detection_summary(summary_df: pd.DataFrame) -> None:
    """Print a compact class-count summary after postprocessing."""

    if summary_df.empty:
        print("Detection summary: no detections", flush=True)
        return

    summary = ", ".join(
        f"{row['class']}: {int(row['count'])} rows/{int(row['detected_frames'])} frames"
        for _, row in summary_df.iterrows()
    )
    print(f"Detection summary: {summary}", flush=True)


def print_team_assignment_summary(summary: Mapping[str, Any]) -> None:
    """Print a compact team assignment summary."""

    teams_found = int(summary.get("teams_found", 0) or 0)
    tracks_sampled = int(summary.get("tracks_sampled", 0) or 0)
    if teams_found >= 2:
        team_1 = summary.get("team_1_color_hex", "")
        team_2 = summary.get("team_2_color_hex", "")
        print(
            f"Team assignment: {teams_found} teams from {tracks_sampled} player tracks "
            f"(team_1={team_1}, team_2={team_2})",
            flush=True,
        )
        return
    note = summary.get("note", "team split unavailable")
    print(f"Team assignment: unavailable from {tracks_sampled} player tracks ({note})", flush=True)


def print_event_measurement_summary(event_df: pd.DataFrame) -> None:
    """Print compact event totals from the event CSV content."""

    if event_df.empty or "row_type" not in event_df.columns:
        print("Event summary: no events measured", flush=True)
        return
    summary_df = event_df[event_df["row_type"].eq("event_summary")]
    if summary_df.empty:
        print("Event summary: no summary rows", flush=True)
        return
    parts = [
        f"{row['event_label_ko']}={int(row['count'])}"
        for _, row in summary_df.iterrows()
        if row.get("count") != ""
    ]
    print(f"Event summary: {', '.join(parts)}", flush=True)


def has_team_specific_classes(model_names: Mapping[int, str]) -> bool:
    """Return whether detector class names appear to encode teams."""

    team_tokens = ("team", "home", "away", "offense", "defense", "white", "dark")
    for raw_name in model_names.values():
        name = str(raw_name).strip().lower()
        if any(token in name for token in team_tokens):
            return True
    return False


def read_video_metadata(video_path: str) -> Dict[str, int | float]:
    """Read fps, width, height, and frame_count from the source video."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        raise RuntimeError("Cannot read video size.")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def build_adaptive_processing_profile(metadata: Mapping[str, int | float]) -> Dict[str, int | float | bool]:
    """입력 영상 메타데이터로 범용 탐지/추적 profile을 만든다.

    목표는 특정 4개 파일명에 맞춘 하드코딩이 아니라, 어떤 경기 영상이 들어와도
    해상도와 FPS 차이 때문에 공 탐지/추적 임계값이 크게 흔들리지 않게 하는 것이다.
    """

    width = float(metadata.get("width", 1280) or 1280)
    height = float(metadata.get("height", 720) or 720)
    fps = float(metadata.get("fps", 30.0) or 30.0)
    max_dim = max(width, height, 1.0)

    # 720p를 기준으로 해상도가 커질수록 pixel 단위 거리 threshold를 함께 키운다.
    # 반대로 FPS가 높으면 한 프레임에서 실제 이동량이 줄어드므로 px/frame threshold를 낮춘다.
    resolution_scale = max(0.85, min(1.45, max_dim / 1280.0))
    fps_scale = max(0.72, min(1.25, 30.0 / fps))
    motion_scale = resolution_scale * fps_scale

    # YOLO 입력 크기는 32의 배수일 때 내부 padding이 깔끔하다.
    # 공은 작아서 일반 객체보다 더 큰 imgsz를 사용하되, 과도한 속도 저하를 막기 위해 상한을 둔다.
    base_imgsz = _round_to_stride(max(1280.0, min(1600.0, max_dim * 1.05)))
    ball_imgsz = _round_to_stride(max(1536.0, min(1920.0, max_dim * 1.25)))
    rescue_imgsz = _round_to_stride(max(1280.0, min(1600.0, max_dim * 1.05)))

    return {
        "imgsz": int(base_imgsz),
        "ball_imgsz": int(ball_imgsz),
        "hoop_imgsz": int(rescue_imgsz),
        "rim_rescue_imgsz": int(rescue_imgsz),
        "ball_rescue_imgsz": int(rescue_imgsz),
        "non_ball_conf_threshold": 0.10,
        "player_rescue_conf_threshold": 0.045,
        "hoop_conf_threshold": 0.035,
        "max_det": 420,
        "rim_rescue_margin_px": 220.0 * resolution_scale,
        "enable_tile_rescue": True,
        "court_ball_margin_px": 320.0 * resolution_scale,
        "court_hoop_margin_px": 360.0 * resolution_scale,
        "rim_proximity_px": 190.0 * resolution_scale,
        "rim_interpolation_extra_gap": 24 if fps >= 45.0 else 18,
        "association_max_distance_px": 220.0 * motion_scale,
        "association_gap_growth_px": 35.0 * motion_scale,
        "association_restart_after_frames": 12 if fps >= 45.0 else 10,
        "max_gap": 16 if fps >= 45.0 else 12,
        "max_prediction_frames": 16 if fps >= 45.0 else 12,
    }


def _round_to_stride(value: float, stride: int = 32) -> int:
    """YOLO 입력 크기를 stride 배수로 반올림한다."""

    return int(round(float(value) / float(stride)) * stride)


def derive_refinement_output_prefix(csv_output_path: str) -> str:
    """tracking_results.csv 이름에서 raw/cleaned/verification 파일 prefix를 가져온다.

    예: A_tracking_results.csv -> A_raw_detection_verify.csv
    단일 영상 기본 실행은 prefix가 빈 문자열이어서 기존 파일명을 유지한다.
    """

    name = Path(csv_output_path).name
    suffix = "tracking_results.csv"
    if name == suffix:
        return ""
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return f"{Path(name).stem}_"


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """Create an MP4 writer that keeps the original fps and frame size."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")
    return writer


def resolve_processing_frame_limit(
    fps: float,
    max_duration_seconds: Optional[float],
    test_mode: bool,
    max_test_frames: int,
) -> Optional[int]:
    """Resolve the frame limit from the duration and test-mode options."""

    limits: List[int] = []
    if max_duration_seconds is not None and max_duration_seconds > 0:
        limits.append(max(1, int(round(float(max_duration_seconds) * float(fps)))))
    if test_mode:
        limits.append(max(1, int(max_test_frames)))
    if not limits:
        return None
    return min(limits)


def should_update_court_polygon(
    frame_id: int,
    interval_frames: int,
    current_polygon: Optional[Sequence[Tuple[float, float]]],
) -> bool:
    """Return whether the court pose model should refresh the court polygon."""

    interval = max(1, int(interval_frames))
    return current_polygon is None or frame_id % interval == 0


def detect_court_polygon(
    frame: Any,
    model: YOLO,
    frame_size: Tuple[int, int],
    device: str,
    use_half: bool,
    confidence_threshold: float,
    keypoint_conf_threshold: float,
    imgsz: int,
    expand_px: float,
) -> Optional[List[Tuple[float, float]]]:
    """Detect the visible court as a pixel polygon from a YOLO pose model."""

    try:
        results = model.predict(
            frame,
            conf=confidence_threshold,
            imgsz=imgsz,
            max_det=1,
            device=device,
            half=use_half,
            verbose=False,
        )
    except Exception as exc:
        print(f"Court keypoint prediction failed: {exc}", flush=True)
        return None

    if not results:
        return None

    result = results[0]
    polygon = court_polygon_from_keypoints(
        result,
        frame_size=frame_size,
        keypoint_conf_threshold=keypoint_conf_threshold,
        expand_px=expand_px,
    )
    if polygon is not None:
        return polygon
    return court_polygon_from_box(result, frame_size=frame_size, expand_px=expand_px)


def court_polygon_from_keypoints(
    result: Any,
    frame_size: Tuple[int, int],
    keypoint_conf_threshold: float,
    expand_px: float,
    min_area_ratio: float = 0.04,
) -> Optional[List[Tuple[float, float]]]:
    """Convert pose keypoints into a convex court polygon."""

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None:
        return None

    xy_instances = keypoints.xy.cpu().tolist()
    conf_tensor = getattr(keypoints, "conf", None)
    conf_instances = conf_tensor.cpu().tolist() if conf_tensor is not None else None
    width, height = frame_size
    min_area = float(width * height) * float(min_area_ratio)

    best_polygon: Optional[List[Tuple[float, float]]] = None
    best_area = 0.0

    for instance_index, points in enumerate(xy_instances):
        confidences = (
            conf_instances[instance_index]
            if conf_instances is not None
            else [1.0] * len(points)
        )
        valid_points: List[Tuple[float, float]] = []
        for point, confidence in zip(points, confidences):
            if float(confidence) < float(keypoint_conf_threshold):
                continue
            if len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
            if not point_is_in_frame((x, y), frame_size):
                continue
            if x == 0.0 and y == 0.0:
                continue
            valid_points.append((x, y))

        hull = convex_hull(valid_points)
        if len(hull) < 3:
            continue

        area = abs(polygon_area(hull))
        if area >= min_area and area > best_area:
            best_area = area
            best_polygon = hull

    if best_polygon is None:
        return None
    return expand_polygon(best_polygon, frame_size=frame_size, expand_px=expand_px)


def court_polygon_from_box(
    result: Any,
    frame_size: Tuple[int, int],
    expand_px: float,
) -> Optional[List[Tuple[float, float]]]:
    """Fallback to the court detector bbox when keypoints are too sparse."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = map(float, best_box.xyxy[0])
    polygon = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    return expand_polygon(polygon, frame_size=frame_size, expand_px=expand_px)


def filter_records_by_court_polygon(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
    court_polygon: Optional[Sequence[Tuple[float, float]]],
    margin_by_class: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Keep only detections inside the learned court polygon."""

    if not records:
        return []

    polygon = court_polygon if court_polygon is not None else DEFAULT_BROADCAST_COURT_POLYGON_NORM
    polygon_is_normalized = court_polygon is None
    try:
        court_filter = CourtObjectFilter(
            polygon,
            frame_size=frame_size,
            polygon_is_normalized=polygon_is_normalized,
            margin_px=12.0,
            margin_by_class=margin_by_class,
        )
    except ValueError:
        return [dict(record) for record in records]

    return court_filter.filter_records(records)


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return points on the convex hull using the monotonic chain algorithm."""

    unique_points = sorted({(round(float(x), 3), round(float(y), 3)) for x, y in points})
    if len(unique_points) <= 1:
        return list(unique_points)

    def cross(
        origin: Tuple[float, float],
        left: Tuple[float, float],
        right: Tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: List[Tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def expand_polygon(
    polygon: Sequence[Tuple[float, float]],
    frame_size: Tuple[int, int],
    expand_px: float,
) -> List[Tuple[float, float]]:
    """Expand a polygon from its centroid and clamp it to the frame."""

    if not polygon:
        return []

    center_x = sum(point[0] for point in polygon) / float(len(polygon))
    center_y = sum(point[1] for point in polygon) / float(len(polygon))
    expanded: List[Tuple[float, float]] = []

    for x, y in polygon:
        dx = float(x) - center_x
        dy = float(y) - center_y
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            expanded.append(clamp_point((x, y), frame_size))
            continue
        scale = (distance + float(expand_px)) / distance
        expanded.append(clamp_point((center_x + dx * scale, center_y + dy * scale), frame_size))

    return expanded


def polygon_area(polygon: Sequence[Tuple[float, float]]) -> float:
    """Return signed polygon area."""

    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def point_is_in_frame(point: Tuple[float, float], frame_size: Tuple[int, int]) -> bool:
    width, height = frame_size
    x, y = point
    return -1.0 <= x <= float(width) + 1.0 and -1.0 <= y <= float(height) + 1.0


def clamp_point(point: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = frame_size
    x, y = point
    return (
        max(0.0, min(float(width), float(x))),
        max(0.0, min(float(height), float(y))),
    )


def extract_detection_records(
    results: Sequence[Any],
    model_names: Mapping[int, str],
    frame_id: int,
    forced_track_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract project target detections from YOLO Results objects."""

    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    frame_records: List[Dict[str, Any]] = []
    for box in boxes:
        cls_id = int(box.cls[0])
        raw_class_name = str(model_names.get(cls_id, cls_id))
        class_name = normalize_class_name(raw_class_name)
        if class_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        confidence = float(box.conf[0])

        track_id = -1
        if forced_track_id is not None:
            track_id = forced_track_id
        elif box.id is not None:
            track_id = int(box.id[0])

        frame_records.append(
            {
                "frame": frame_id,
                "track_id": track_id,
                "class": class_name,
                "confidence": round(confidence, 3),
                "x_center": round((x1 + x2) / 2.0, 2),
                "y_center": round((y1 + y2) / 2.0, 2),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )

    return frame_records


def detect_ball_records(
    frame: Any,
    model: YOLO,
    model_names: Mapping[int, str],
    frame_id: int,
    frame_size: Tuple[int, int],
    ball_class_ids: Sequence[int],
    context_records: Sequence[Mapping[str, Any]],
    device: str,
    use_half: bool,
    ball_conf_threshold: float,
    ball_iou_threshold: float,
    ball_imgsz: int,
    ball_max_det: int,
    enable_rim_ball_rescue: bool,
    rim_rescue_conf_threshold: float,
    rim_rescue_iou_threshold: float,
    rim_rescue_margin_px: float,
    rim_rescue_imgsz: int,
    enable_tile_rescue: bool,
    rescue_conf_threshold: float,
    rescue_iou_threshold: float,
    rescue_imgsz: int,
    model_label: str = "ball_model",
    model_priority_bonus: float = 0.0,
) -> List[Dict[str, Any]]:
    """공 후보를 검출합니다.

    공은 작고 모션 블러가 심해 일반 객체보다 놓치기 쉽습니다. 그래서 전체 프레임 검출 후,
    필요하면 골대 주변 crop rescue 또는 화면 타일 crop rescue를 추가로 수행합니다.
    반환되는 후보는 아직 최종 공 1개가 아니며, ball_tracker.py에서 궤적 기준으로 다시 선택됩니다.
    """

    if not ball_class_ids:
        return []

    ball_results = model.predict(
        frame,
        conf=ball_conf_threshold,
        iou=ball_iou_threshold,
        imgsz=ball_imgsz,
        classes=list(ball_class_ids),
        max_det=ball_max_det,
        device=device,
        half=use_half,
        verbose=False,
    )
    ball_records = extract_detection_records(
        ball_results,
        model_names,
        frame_id,
        forced_track_id=-1,
    )
    ball_records = annotate_ball_model_records(ball_records, model_label, model_priority_bonus)
    kept_records = filter_ball_candidates(
        ball_records,
        frame_size,
        context_records=context_records,
    )

    if enable_rim_ball_rescue and not kept_records:
        rim_crops = make_rim_rescue_crops(
            frame_size=frame_size,
            context_records=context_records,
            margin_px=rim_rescue_margin_px,
        )
        rim_records = detect_ball_records_in_crops(
            frame=frame,
            model=model,
            model_names=model_names,
            frame_id=frame_id,
            ball_class_ids=ball_class_ids,
            crops=rim_crops,
            device=device,
            use_half=use_half,
            confidence_threshold=rim_rescue_conf_threshold,
            iou_threshold=rim_rescue_iou_threshold,
            imgsz=rim_rescue_imgsz,
            max_det=max(12, ball_max_det // 3),
        )
        rim_records = annotate_ball_model_records(rim_records, model_label, model_priority_bonus)
        if rim_records:
            return filter_ball_candidates(
                rim_records,
                frame_size,
                context_records=context_records,
            )

    # crop rescue는 추가 YOLO 호출이 필요하므로 기본값은 꺼져 있다.
    # full-frame에서 공 후보가 하나라도 나오면 바로 반환해 불필요한 추론을 막는다.
    if kept_records or not enable_tile_rescue or not context_records:
        return kept_records

    rescue_records: List[Dict[str, Any]] = []
    for crop in make_ball_rescue_crops(frame_size):
        x1, y1, x2, y2 = crop
        crop_frame = frame[y1:y2, x1:x2]
        if crop_frame.size == 0:
            continue

        crop_records = detect_ball_records_in_crops(
            frame=frame,
            model=model,
            model_names=model_names,
            frame_id=frame_id,
            ball_class_ids=ball_class_ids,
            crops=[crop],
            device=device,
            use_half=use_half,
            confidence_threshold=rescue_conf_threshold,
            iou_threshold=rescue_iou_threshold,
            imgsz=rescue_imgsz,
            max_det=max(20, ball_max_det // 2),
        )
        crop_records = annotate_ball_model_records(crop_records, model_label, model_priority_bonus)
        crop_records = attenuate_rescue_ball_confidence(crop_records)
        rescue_records.extend(crop_records)

    if not rescue_records:
        return kept_records

    rescue_records = filter_rescue_ball_candidates_by_context(
        rescue_records,
        context_records=context_records,
        frame_size=frame_size,
    )
    if not rescue_records:
        return kept_records

    return filter_ball_candidates(
        kept_records + rescue_records,
        frame_size,
        context_records=context_records,
    )


def has_class_record(records: Sequence[Mapping[str, Any]], class_name: str) -> bool:
    """현재 프레임 탐지 결과에 특정 클래스가 이미 있는지 확인한다."""

    return any(str(record.get("class")) == class_name for record in records)


def count_class_records(records: Sequence[Mapping[str, Any]], class_name: str) -> int:
    """Count records for a normalized class name."""

    return sum(1 for record in records if str(record.get("class")) == class_name)


def should_use_secondary_model(
    primary_model_path: Optional[str],
    secondary_model_path: Optional[str],
) -> bool:
    """Return whether a distinct secondary ball model should be loaded."""

    if not secondary_model_path:
        return False
    if not primary_model_path:
        return True
    try:
        return Path(primary_model_path).resolve() != Path(secondary_model_path).resolve()
    except OSError:
        return str(primary_model_path) != str(secondary_model_path)


def annotate_ball_model_records(
    records: Sequence[Mapping[str, Any]],
    model_label: str,
    model_priority_bonus: float,
) -> List[Dict[str, Any]]:
    """Attach detector-source metadata used before CSV column narrowing."""

    annotated: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["ball_model_source"] = model_label
        row["ball_model_sources"] = model_label
        row["ball_model_priority_bonus"] = float(model_priority_bonus)
        row["model_agreement_count"] = 1
        annotated.append(row)
    return annotated


def detect_player_records(
    frame: Any,
    model: YOLO,
    model_names: Mapping[int, str],
    frame_id: int,
    frame_size: Tuple[int, int],
    player_class_ids: Sequence[int],
    device: str,
    use_half: bool,
    confidence_threshold: float,
    iou_threshold: float,
    imgsz: int,
    max_det: int,
) -> List[Dict[str, Any]]:
    """Run a lower-confidence player-only pass for missed visible players."""

    if not player_class_ids:
        return []

    player_results = model.predict(
        frame,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        classes=list(player_class_ids),
        max_det=max_det,
        device=device,
        half=use_half,
        verbose=False,
    )
    player_records = extract_detection_records(
        player_results,
        model_names,
        frame_id,
        forced_track_id=-1,
    )
    return filter_player_candidates(player_records, frame_size)


def merge_player_records(
    base_records: Sequence[Mapping[str, Any]],
    rescue_records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Merge player rescue boxes without duplicating tracked players."""

    if not rescue_records:
        return [dict(record) for record in base_records]

    non_player_records = [
        dict(record)
        for record in base_records
        if str(record.get("class")) != "player"
    ]
    player_records = [
        dict(record)
        for record in [*base_records, *rescue_records]
        if str(record.get("class")) == "player"
    ]
    return non_player_records + deduplicate_player_candidates(
        filter_player_candidates(player_records, frame_size),
        frame_size,
    )


def detect_hoop_records(
    frame: Any,
    model: YOLO,
    model_names: Mapping[int, str],
    frame_id: int,
    frame_size: Tuple[int, int],
    hoop_class_ids: Sequence[int],
    device: str,
    use_half: bool,
    confidence_threshold: float,
    iou_threshold: float,
    imgsz: int,
    max_det: int,
) -> List[Dict[str, Any]]:
    """링/골대 객체를 낮은 confidence로 한 번 더 탐지한다.

    일반 ByteTrack pass는 선수 ID 안정성을 위해 confidence를 너무 낮출 수 없다.
    반면 링은 작고 화면 상단/측면에 있어 놓치기 쉬우므로, 링이 없는 프레임에서만
    별도 pass를 수행해 후보를 보강한다. 후보는 크기/위치/비율 필터를 거쳐 false positive를 줄인다.
    """

    if not hoop_class_ids:
        return []

    hoop_results = model.predict(
        frame,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        classes=list(hoop_class_ids),
        max_det=max_det,
        device=device,
        half=use_half,
        verbose=False,
    )
    hoop_records = extract_detection_records(
        hoop_results,
        model_names,
        frame_id,
        forced_track_id=-1,
    )
    return filter_hoop_candidates(hoop_records, frame_size)


def merge_hoop_records(
    base_records: Sequence[Mapping[str, Any]],
    rescue_records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """기존 프레임 결과와 링 rescue 결과를 합치고 중복 링 bbox를 제거한다."""

    if not rescue_records:
        return [dict(record) for record in base_records]

    non_hoop_records = [
        dict(record)
        for record in base_records
        if str(record.get("class")) != "hoop"
    ]
    hoop_records = [
        dict(record)
        for record in [*base_records, *rescue_records]
        if str(record.get("class")) == "hoop"
    ]
    return non_hoop_records + filter_hoop_candidates(hoop_records, frame_size)


def filter_hoop_candidates(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """링 후보의 크기/위치/비율을 검사해 명백한 오탐을 제거한다.

    링은 공보다 크지만 선수 bbox나 광고판보다 훨씬 작고, 보통 화면 중상단에 위치한다.
    이 조건을 하드코딩 픽셀이 아니라 해상도 scale로 보정해 여러 영상에 대응한다.
    """

    width, height = frame_size
    frame_area = float(width * height)
    scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    kept: List[Dict[str, Any]] = []

    for record in records:
        if str(record.get("class")) != "hoop" or not _record_has_bbox(record):
            continue

        box_width = float(record["x2"]) - float(record["x1"])
        box_height = float(record["y2"]) - float(record["y1"])
        if box_width <= 4.0 * scale or box_height <= 4.0 * scale:
            continue

        area = box_width * box_height
        aspect_ratio = box_width / max(1.0, box_height)
        x_center = float(record["x_center"])
        y_center = float(record["y_center"])

        # 링이 화면 아래쪽에 잡히는 경우는 관중석/유니폼/광고판 오탐일 확률이 높다.
        # 단, 카메라 구도가 다른 영상도 고려해 78% 지점까지만 느슨하게 허용한다.
        if y_center > height * 0.78:
            continue
        if x_center < -width * 0.03 or x_center > width * 1.03:
            continue
        if area > frame_area * 0.035:
            continue
        if max(box_width, box_height) > max(width, height) * 0.23:
            continue
        if aspect_ratio < 0.25 or aspect_ratio > 4.5:
            continue

        kept.append(dict(record))

    kept.sort(key=lambda record: _hoop_candidate_quality(record, frame_size), reverse=True)
    return deduplicate_hoop_candidates(kept, frame_size)[:4]


def deduplicate_hoop_candidates(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """동일 링이 여러 pass에서 중복 검출된 경우 하나만 남긴다."""

    scale = max(0.85, min(1.45, max(frame_size) / 1280.0))
    deduped: List[Dict[str, Any]] = []
    for record in records:
        duplicate = False
        for kept in deduped:
            if _ball_candidate_overlap(record, kept) > 0.35:
                duplicate = True
                break
            center_distance = math.hypot(
                float(record["x_center"]) - float(kept["x_center"]),
                float(record["y_center"]) - float(kept["y_center"]),
            )
            if center_distance <= 28.0 * scale:
                duplicate = True
                break
        if not duplicate:
            deduped.append(dict(record))
    return deduped


def _hoop_candidate_quality(record: Mapping[str, Any], frame_size: Tuple[int, int]) -> float:
    """링 후보를 confidence, bbox 모양, 화면 위치 기준으로 점수화한다."""

    width, height = frame_size
    confidence = float(record.get("confidence", 0.0) or 0.0)
    box_width = max(1.0, float(record["x2"]) - float(record["x1"]))
    box_height = max(1.0, float(record["y2"]) - float(record["y1"]))
    area_ratio = (box_width * box_height) / max(1.0, float(width * height))
    aspect_ratio = box_width / box_height
    y_center = float(record["y_center"])

    aspect_penalty = abs(1.25 - aspect_ratio) * 0.025
    area_penalty = max(0.0, area_ratio - 0.006) * 6.0
    upper_bonus = max(0.0, (height * 0.65 - y_center) / max(1.0, height)) * 0.04
    return confidence + upper_bonus - aspect_penalty - area_penalty


def filter_player_candidates(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Keep plausible player boxes from the low-confidence rescue pass."""

    width, height = frame_size
    frame_area = float(width * height)
    scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    kept: List[Dict[str, Any]] = []

    for record in records:
        if str(record.get("class")) != "player" or not _record_has_bbox(record):
            continue

        box_width = float(record["x2"]) - float(record["x1"])
        box_height = float(record["y2"]) - float(record["y1"])
        if box_width <= 10.0 * scale or box_height <= 22.0 * scale:
            continue
        if box_height > height * 0.95 or box_width > width * 0.38:
            continue

        area = box_width * box_height
        if area < frame_area * 0.00018 or area > frame_area * 0.22:
            continue

        aspect_ratio = box_width / max(1.0, box_height)
        if aspect_ratio < 0.12 or aspect_ratio > 1.35:
            continue

        kept.append(dict(record))

    kept.sort(key=_player_candidate_quality, reverse=True)
    return kept[:18]


def deduplicate_player_candidates(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Remove duplicated player boxes while keeping stronger tracked boxes."""

    scale = max(0.85, min(1.45, max(frame_size) / 1280.0))
    deduped: List[Dict[str, Any]] = []
    for record in records:
        duplicate = False
        for kept in deduped:
            if _ball_candidate_overlap(record, kept) > 0.42:
                duplicate = True
                break
            center_distance = math.hypot(
                float(record["x_center"]) - float(kept["x_center"]),
                float(record["y_center"]) - float(kept["y_center"]),
            )
            if center_distance <= 32.0 * scale:
                duplicate = True
                break
        if not duplicate:
            deduped.append(dict(record))
    return deduped[:14]


def _player_candidate_quality(record: Mapping[str, Any]) -> float:
    confidence = float(record.get("confidence", 0.0) or 0.0)
    try:
        track_id = int(float(record.get("track_id", -1) or -1))
    except (TypeError, ValueError):
        track_id = -1
    tracked_bonus = 0.10 if track_id >= 0 else 0.0
    box_width = max(1.0, float(record["x2"]) - float(record["x1"]))
    box_height = max(1.0, float(record["y2"]) - float(record["y1"]))
    aspect_ratio = box_width / box_height
    aspect_penalty = abs(0.42 - aspect_ratio) * 0.035
    return confidence + tracked_bonus - aspect_penalty


def make_ball_rescue_crops(frame_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Create overlapping horizontal crops that enlarge small ball candidates."""

    width, height = frame_size
    left_end = int(round(width * 0.58))
    right_start = int(round(width * 0.42))
    center_start = int(round(width * 0.20))
    center_end = int(round(width * 0.80))

    crops = [
        (0, 0, left_end, height),
        (right_start, 0, width, height),
        (center_start, 0, center_end, height),
    ]
    return [
        (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        for x1, y1, x2, y2 in crops
        if x2 - x1 >= 64 and y2 - y1 >= 64
    ]


def make_rim_rescue_crops(
    frame_size: Tuple[int, int],
    context_records: Sequence[Mapping[str, Any]],
    margin_px: float,
) -> List[Tuple[int, int, int, int]]:
    """Create low-threshold rescue crops around detected hoop/rim boxes."""

    crops: List[Tuple[int, int, int, int]] = []
    for record in context_records:
        if str(record.get("class")) != "hoop" or not _record_has_bbox(record):
            continue
        x1 = float(record["x1"]) - float(margin_px)
        y1 = float(record["y1"]) - float(margin_px)
        x2 = float(record["x2"]) + float(margin_px)
        y2 = float(record["y2"]) + float(margin_px)
        crop = clamp_crop((x1, y1, x2, y2), frame_size)
        if crop is not None:
            crops.append(crop)
    return deduplicate_crops(crops)


def detect_ball_records_in_crops(
    frame: Any,
    model: YOLO,
    model_names: Mapping[int, str],
    frame_id: int,
    ball_class_ids: Sequence[int],
    crops: Sequence[Tuple[int, int, int, int]],
    device: str,
    use_half: bool,
    confidence_threshold: float,
    iou_threshold: float,
    imgsz: int,
    max_det: int,
) -> List[Dict[str, Any]]:
    """Run a lower-threshold ball detector in selected crops."""

    crop_records: List[Dict[str, Any]] = []
    for crop in crops:
        x1, y1, x2, y2 = crop
        crop_frame = frame[y1:y2, x1:x2]
        if crop_frame.size == 0:
            continue

        crop_results = model.predict(
            crop_frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            classes=list(ball_class_ids),
            max_det=max_det,
            device=device,
            half=use_half,
            verbose=False,
        )
        records = extract_detection_records(
            crop_results,
            model_names,
            frame_id,
            forced_track_id=-1,
        )
        crop_records.extend(offset_detection_records(records, x_offset=x1, y_offset=y1))
    return crop_records


def clamp_crop(
    crop: Tuple[float, float, float, float],
    frame_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a crop box to the frame and discard boxes that are too small."""

    width, height = frame_size
    x1, y1, x2, y2 = crop
    output = (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )
    if output[2] - output[0] < 64 or output[3] - output[1] < 64:
        return None
    return output


def deduplicate_crops(
    crops: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Remove identical crop boxes while preserving order."""

    seen = set()
    output: List[Tuple[int, int, int, int]] = []
    for crop in crops:
        if crop in seen:
            continue
        seen.add(crop)
        output.append(crop)
    return output


def offset_detection_records(
    records: Sequence[Mapping[str, Any]],
    x_offset: float,
    y_offset: float,
) -> List[Dict[str, Any]]:
    """Move crop-local detections back into full-frame coordinates."""

    adjusted: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for column in ["x_center", "x1", "x2"]:
            row[column] = round(float(row[column]) + float(x_offset), 2)
        for column in ["y_center", "y1", "y2"]:
            row[column] = round(float(row[column]) + float(y_offset), 2)
        adjusted.append(row)
    return adjusted


def attenuate_rescue_ball_confidence(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Lower crop-rescue confidence so it fills gaps without starting false tracks."""

    adjusted: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        row["confidence"] = round(min(0.07, confidence * 0.35), 3)
        adjusted.append(row)
    return adjusted


def filter_rescue_ball_candidates_by_context(
    records: Sequence[Mapping[str, Any]],
    context_records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Keep rescue candidates only when they are near basketball context objects."""

    width, height = frame_size
    scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    context = [
        record
        for record in context_records
        if str(record.get("class")) in {"player", "referee", "hoop"}
        and _record_has_bbox(record)
    ]
    if not context:
        return []

    filtered: List[Dict[str, Any]] = []
    for record in records:
        ball_x = float(record["x_center"])
        ball_y = float(record["y_center"])
        if ball_y < height * 0.08:
            continue

        for context_record in context:
            class_name = str(context_record.get("class"))
            margin = (250.0 if class_name == "hoop" else 185.0) * scale
            if _point_inside_expanded_bbox(ball_x, ball_y, context_record, margin):
                filtered.append(dict(record))
                break

    return filtered


def filter_ball_candidates(
    records: Sequence[Mapping[str, Any]],
    frame_size: Tuple[int, int],
    context_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """명백히 잘못된 공 bbox를 제거하고 프레임당 후보 수를 제한합니다.

    여기서는 공답지 않은 크기/비율/중복 후보만 줄입니다. 진짜 공을 너무 일찍 버리지 않기 위해
    낮은 confidence 후보도 일부 남기고, 최종 선택은 ball_tracker.py의 motion association에 맡깁니다.
    """

    width, height = frame_size
    frame_area = float(width * height)
    resolution_scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    max_ball_area = min(frame_area * 0.0032, 5200.0 * resolution_scale * resolution_scale)
    max_ball_side = 95.0 * resolution_scale
    kept: List[Dict[str, Any]] = []

    for record in records:
        if record.get("class") != "ball":
            continue

        box_width = float(record["x2"]) - float(record["x1"])
        box_height = float(record["y2"]) - float(record["y1"])
        if box_width <= 3.0 or box_height <= 3.0:
            continue

        area = box_width * box_height
        aspect_ratio = box_width / max(1.0, box_height)

        # The ball should be compact. This rejects large scoreboard/overlay mistakes.
        if area > max_ball_area:
            continue
        if max(box_width, box_height) > max_ball_side:
            continue
        if aspect_ratio < 0.45 or aspect_ratio > 2.2:
            continue
        if _is_likely_rim_false_ball(record, context_records, frame_size):
            continue

        kept.append(dict(record))

    # Keep only the strongest compact candidates per frame. This reduces random
    # low-confidence false balls before the motion association step.
    kept.sort(key=_ball_candidate_quality, reverse=True)
    return deduplicate_ball_candidates(kept)[:12]


def deduplicate_ball_candidates(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate ball boxes produced by overlapping rescue crops."""

    deduped: List[Dict[str, Any]] = []
    for record in records:
        duplicate = False
        for kept_index, kept in enumerate(deduped):
            if _ball_candidate_overlap(record, kept) > 0.45:
                deduped[kept_index] = merge_duplicate_ball_candidate(record, kept)
                duplicate = True
                break
            center_distance = math.hypot(
                float(record["x_center"]) - float(kept["x_center"]),
                float(record["y_center"]) - float(kept["y_center"]),
            )
            if center_distance <= max(
                10.0,
                _ball_candidate_size(record) * 0.65,
                _ball_candidate_size(kept) * 0.65,
            ):
                deduped[kept_index] = merge_duplicate_ball_candidate(record, kept)
                duplicate = True
                break
        if not duplicate:
            deduped.append(dict(record))
    return deduped


def merge_duplicate_ball_candidate(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge two overlapping ball candidates, keeping the better bbox."""

    first_sources = _candidate_sources(first)
    second_sources = _candidate_sources(second)
    combined_sources = sorted(first_sources | second_sources)

    best = dict(first if _ball_candidate_quality(first) >= _ball_candidate_quality(second) else second)
    first_confidence = float(first.get("confidence", 0.0) or 0.0)
    second_confidence = float(second.get("confidence", 0.0) or 0.0)
    agreement_bonus = 0.04 * max(0, len(combined_sources) - 1)
    best["confidence"] = round(min(0.99, max(first_confidence, second_confidence) + agreement_bonus), 3)
    best["ball_model_sources"] = "|".join(combined_sources)
    best["ball_model_source"] = best.get("ball_model_source") or combined_sources[0]
    best["model_agreement_count"] = len(combined_sources)
    best["ball_model_priority_bonus"] = max(
        float(first.get("ball_model_priority_bonus", 0.0) or 0.0),
        float(second.get("ball_model_priority_bonus", 0.0) or 0.0),
    )
    return best


def _candidate_sources(record: Mapping[str, Any]) -> Set[str]:
    sources_value = record.get("ball_model_sources") or record.get("ball_model_source") or ""
    sources = {
        source.strip()
        for source in str(sources_value).split("|")
        if source and source.strip()
    }
    if not sources:
        sources.add("unknown")
    return sources


def _ball_candidate_quality(record: Mapping[str, Any]) -> float:
    """Score a ball candidate by confidence, compactness, and reasonable size."""

    confidence = float(record.get("confidence", 0.0) or 0.0)
    width = float(record["x2"]) - float(record["x1"])
    height = float(record["y2"]) - float(record["y1"])
    area = max(1.0, width * height)
    aspect_ratio = width / max(1.0, height)

    aspect_penalty = abs(1.0 - aspect_ratio) * 0.08
    # Broadcast basketball boxes are usually small. Very large boxes are suspicious.
    size_penalty = max(0.0, area - 1600.0) / 8000.0
    agreement_bonus = 0.035 * max(0, int(record.get("model_agreement_count", 1) or 1) - 1)
    source_bonus = float(record.get("ball_model_priority_bonus", 0.0) or 0.0)
    return confidence + agreement_bonus + source_bonus - aspect_penalty - size_penalty


def _is_likely_rim_false_ball(
    record: Mapping[str, Any],
    context_records: Optional[Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
) -> bool:
    """Reject low-confidence ball candidates that are probably the rim itself."""

    if not context_records:
        return False

    confidence = float(record.get("confidence", 0.0) or 0.0)
    ball_size = _ball_candidate_size(record)
    ball_area = _candidate_area(record)
    if ball_area <= 0.0:
        return True

    width, height = frame_size
    scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    ball_x = float(record["x_center"])
    ball_y = float(record["y_center"])

    for context in context_records:
        if str(context.get("class")) != "hoop" or not _record_has_bbox(context):
            continue

        hoop_width = float(context["x2"]) - float(context["x1"])
        hoop_height = float(context["y2"]) - float(context["y1"])
        if hoop_width <= 6.0 or hoop_height <= 6.0:
            continue

        hoop_size = max(hoop_width, hoop_height)
        center_inside_hoop = _point_inside_expanded_bbox(ball_x, ball_y, context, 3.0 * scale)
        ball_inside_ratio = _candidate_intersection_ratio(record, context)
        near_hoop_center = math.hypot(
            ball_x - float(context["x_center"]),
            ball_y - float(context["y_center"]),
        ) <= max(18.0 * scale, hoop_size * 0.28)

        # The common false positive is a tiny, low-confidence ball box stuck on
        # the rim/hoop bbox. A real shot can still pass near the rim, so this is
        # intentionally limited to weak candidates mostly inside the hoop box.
        if (
            confidence < 0.09
            and center_inside_hoop
            and ball_inside_ratio >= 0.70
            and ball_size <= max(42.0 * scale, hoop_size * 0.48)
        ):
            return True
        if (
            confidence < 0.16
            and near_hoop_center
            and ball_inside_ratio >= 0.88
            and ball_size <= max(54.0 * scale, hoop_size * 0.62)
        ):
            return True

    return False


def _record_has_bbox(record: Mapping[str, Any]) -> bool:
    return all(
        _as_finite_float(record.get(column)) is not None
        for column in ["x1", "y1", "x2", "y2"]
    )


def _point_inside_expanded_bbox(
    x: float,
    y: float,
    record: Mapping[str, Any],
    margin: float,
) -> bool:
    x1 = float(record["x1"]) - margin
    y1 = float(record["y1"]) - margin
    x2 = float(record["x2"]) + margin
    y2 = float(record["y2"]) + margin
    return x1 <= x <= x2 and y1 <= y <= y2


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _ball_candidate_size(record: Mapping[str, Any]) -> float:
    """Return the larger side of a ball candidate box."""

    width = float(record["x2"]) - float(record["x1"])
    height = float(record["y2"]) - float(record["y1"])
    return max(width, height)


def _ball_candidate_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Compute IoU between two candidate boxes."""

    first_x1, first_y1, first_x2, first_y2 = _candidate_bbox(first)
    second_x1, second_y1, second_x2, second_y2 = _candidate_bbox(second)

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _candidate_intersection_ratio(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    """Return how much of the first bbox is covered by the second bbox."""

    first_x1, first_y1, first_x2, first_y2 = _candidate_bbox(first)
    second_x1, second_y1, second_x2, second_y2 = _candidate_bbox(second)
    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    first_area = _candidate_area(first)
    if first_area <= 0.0:
        return 0.0
    return intersection / first_area


def _candidate_area(record: Mapping[str, Any]) -> float:
    x1, y1, x2, y2 = _candidate_bbox(record)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _candidate_bbox(record: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(record["x1"]),
        float(record["y1"]),
        float(record["x2"]),
        float(record["y2"]),
    )


def normalize_class_name(raw_name: str) -> str:
    """Normalize model class names into project class names."""

    normalized = raw_name.strip().lower()
    return CLASS_ALIASES.get(normalized, normalized)


def resolve_class_ids(model_names: Mapping[int, str], wanted_classes: Set[str]) -> List[int]:
    """Find YOLO class ids that match wanted project class names."""

    class_ids: List[int] = []
    for class_id, raw_name in model_names.items():
        if normalize_class_name(str(raw_name)) in wanted_classes:
            class_ids.append(int(class_id))
    return class_ids


def resolve_yolo_device(device: Optional[str] = None) -> str:
    """Use the requested device, or automatically choose CUDA GPU when available."""

    if device:
        normalized = str(device).strip().lower()
        if normalized in {"cuda", "gpu"}:
            return "0"
        if normalized.startswith("cuda:"):
            return normalized.split(":", 1)[1] or "0"
        return normalized

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "0"
    return "cpu"


def make_tracking_dataframe(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert detection records into a stable DataFrame shape."""

    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    if df.empty:
        return df

    return df.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)


def make_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the required CSV columns and fill missing bbox values."""

    output_df = df.copy()
    for column in CSV_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = None

    output_df = fill_missing_bbox_from_center(output_df)
    output_df = output_df[CSV_COLUMNS]

    if not output_df.empty:
        output_df = output_df.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    return output_df


def fill_missing_bbox_from_center(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small bbox when a corrected ball row only has center coordinates."""

    output_df = df.copy()
    default_box_size = {
        "ball": 18.0,
        "player": 70.0,
        "referee": 70.0,
        "hoop": 80.0,
    }

    for index, row in output_df.iterrows():
        has_bbox = (
            pd.notna(row.get("x1"))
            and pd.notna(row.get("y1"))
            and pd.notna(row.get("x2"))
            and pd.notna(row.get("y2"))
        )
        if has_bbox:
            continue

        x_center = row.get("x_center")
        y_center = row.get("y_center")
        if pd.isna(x_center) or pd.isna(y_center):
            continue

        size = default_box_size.get(str(row.get("class")), 40.0)
        half = size / 2.0
        output_df.at[index, "x1"] = round(float(x_center) - half, 2)
        output_df.at[index, "y1"] = round(float(y_center) - half, 2)
        output_df.at[index, "x2"] = round(float(x_center) + half, 2)
        output_df.at[index, "y2"] = round(float(y_center) + half, 2)

    return output_df


def remove_invalid_ball_positions(df: pd.DataFrame, frame_size: Tuple[int, int]) -> pd.DataFrame:
    """Drop corrected ball rows that drift outside the visible frame."""

    if df.empty:
        return df

    width, height = frame_size
    output_df = df.copy()
    is_ball = output_df["class"].eq("ball")
    x = pd.to_numeric(output_df["x_center"], errors="coerce")
    y = pd.to_numeric(output_df["y_center"], errors="coerce")
    inside_frame = x.between(0, width) & y.between(0, height)

    # Non-ball rows are kept. Ball rows are kept only if their center is visible.
    output_df = output_df[~is_ball | inside_frame].reset_index(drop=True)
    return output_df


def remove_rim_false_ball_positions(
    df: pd.DataFrame,
    frame_size: Tuple[int, int],
    fps: float,
) -> pd.DataFrame:
    """Drop ball rows that are likely rim/hoop false positives."""

    if df.empty or "class" not in df.columns:
        return df

    output_df = df.copy()
    numeric_columns = ["frame", "confidence", "x_center", "y_center", "x1", "y1", "x2", "y2"]
    for column in numeric_columns:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")

    ball_mask = output_df["class"].eq("ball")
    hoop_mask = output_df["class"].eq("hoop")
    if not bool(ball_mask.any()):
        return df

    width, height = frame_size
    scale = max(0.85, min(1.45, max(width, height) / 1280.0))
    hoop_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for row in output_df[hoop_mask].dropna(subset=["frame"]).to_dict("records"):
        try:
            frame = int(row["frame"])
        except (TypeError, ValueError):
            continue
        hoop_by_frame.setdefault(frame, []).append(row)
    player_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    player_mask = output_df["class"].eq("player")
    for row in output_df[player_mask].dropna(subset=["frame"]).to_dict("records"):
        try:
            frame = int(row["frame"])
        except (TypeError, ValueError):
            continue
        player_by_frame.setdefault(frame, []).append(row)

    remove_indices: Set[int] = set()
    near_rim_rows: List[Tuple[int, int, float, float, float]] = []
    near_edge_rows: List[Tuple[int, int, float, float, float]] = []
    for index, row in output_df[ball_mask].dropna(subset=["frame"]).iterrows():
        record = row.to_dict()
        try:
            frame = int(record["frame"])
        except (TypeError, ValueError):
            continue
        if edge_static_ball_candidate(record, frame_size, scale):
            near_edge_rows.append(
                (
                    int(index),
                    frame,
                    float(record["x_center"]),
                    float(record["y_center"]),
                    float(record.get("confidence", 0.0) or 0.0),
                )
            )
        hoop = nearest_hoop_record(frame, record, hoop_by_frame, search_window=3)
        if hoop is None:
            continue

        rim_score = rim_false_ball_score(record, hoop, scale)
        if rim_score["near_rim"]:
            near_rim_rows.append(
                (
                    int(index),
                    frame,
                    float(record["x_center"]),
                    float(record["y_center"]),
                    float(rim_score["distance_to_hoop_center"]),
                )
            )
        if rim_score["remove_now"]:
            remove_indices.add(int(index))

    remove_indices.update(static_rim_false_ball_indices(near_rim_rows, scale, fps))
    remove_indices.update(
        non_shot_rim_ball_indices(
            output_df=output_df,
            near_rim_rows=near_rim_rows,
            hoop_by_frame=hoop_by_frame,
            frame_size=frame_size,
            scale=scale,
            fps=fps,
        )
    )
    remove_indices.update(static_edge_false_ball_indices(near_edge_rows, scale, fps))
    remove_indices.update(
        static_contextless_ball_indices(
            output_df=output_df,
            player_by_frame=player_by_frame,
            frame_size=frame_size,
            scale=scale,
            fps=fps,
        )
    )
    if not remove_indices:
        return df

    return output_df.drop(index=sorted(remove_indices)).reset_index(drop=True)


def nearest_hoop_record(
    frame: int,
    ball_record: Mapping[str, Any],
    hoop_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    search_window: int,
) -> Optional[Mapping[str, Any]]:
    """Find the nearest hoop record around a frame."""

    best_hoop: Optional[Mapping[str, Any]] = None
    best_score = float("inf")
    ball_x = _as_finite_float(ball_record.get("x_center"))
    ball_y = _as_finite_float(ball_record.get("y_center"))
    if ball_x is None or ball_y is None:
        return None

    for current_frame in range(frame - search_window, frame + search_window + 1):
        for hoop in hoop_by_frame.get(current_frame, []):
            if not _record_has_bbox(hoop):
                continue
            hoop_x = float(hoop["x_center"])
            hoop_y = float(hoop["y_center"])
            frame_penalty = abs(current_frame - frame) * 18.0
            score = math.hypot(ball_x - hoop_x, ball_y - hoop_y) + frame_penalty
            if score < best_score:
                best_score = score
                best_hoop = hoop
    return best_hoop


def rim_false_ball_score(
    ball_record: Mapping[str, Any],
    hoop_record: Mapping[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Measure whether a ball row looks like the rim itself."""

    confidence = float(ball_record.get("confidence", 0.0) or 0.0)
    ball_x = float(ball_record["x_center"])
    ball_y = float(ball_record["y_center"])
    hoop_width = float(hoop_record["x2"]) - float(hoop_record["x1"])
    hoop_height = float(hoop_record["y2"]) - float(hoop_record["y1"])
    hoop_size = max(hoop_width, hoop_height, 1.0)
    distance = math.hypot(
        ball_x - float(hoop_record["x_center"]),
        ball_y - float(hoop_record["y_center"]),
    )
    inside_ratio = _candidate_intersection_ratio(ball_record, hoop_record)
    center_inside = _point_inside_expanded_bbox(ball_x, ball_y, hoop_record, 5.0 * scale)
    near_center = distance <= max(22.0 * scale, hoop_size * 0.34)
    ball_size = _ball_candidate_size(ball_record)
    small_enough = ball_size <= max(52.0 * scale, hoop_size * 0.62)

    remove_now = (
        confidence < 0.20
        and small_enough
        and ((center_inside and inside_ratio >= 0.50) or (near_center and inside_ratio >= 0.35))
    )
    return {
        "near_rim": bool(small_enough and (center_inside or near_center) and inside_ratio >= 0.25),
        "remove_now": bool(remove_now),
        "distance_to_hoop_center": distance,
    }


def static_rim_false_ball_indices(
    near_rim_rows: Sequence[Tuple[int, int, float, float, float]],
    scale: float,
    fps: float,
) -> Set[int]:
    """Remove near-rim ball rows that stay almost fixed for several frames."""

    if not near_rim_rows:
        return set()

    max_gap = max(2, int(round((float(fps) if fps and fps > 0 else 30.0) * 0.18)))
    min_rows = 3
    max_spread = 14.0 * scale
    remove_indices: Set[int] = set()
    current: List[Tuple[int, int, float, float, float]] = []

    for item in sorted(near_rim_rows, key=lambda row: row[1]):
        if current and item[1] - current[-1][1] > max_gap:
            remove_indices.update(static_rim_run_indices(current, min_rows, max_spread))
            current = []
        current.append(item)

    remove_indices.update(static_rim_run_indices(current, min_rows, max_spread))
    return remove_indices


def non_shot_rim_ball_indices(
    output_df: pd.DataFrame,
    near_rim_rows: Sequence[Tuple[int, int, float, float, float]],
    hoop_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
    scale: float,
    fps: float,
) -> Set[int]:
    """Remove rim-near ball clusters unless they look like shot approaches."""

    if not near_rim_rows:
        return set()

    ball_df = output_df[output_df["class"].eq("ball")].copy()
    if ball_df.empty:
        return set()

    max_gap = max(3, int(round((float(fps) if fps and fps > 0 else 30.0) * 0.45)))
    remove_indices: Set[int] = set()
    for cluster in cluster_indexed_frame_rows(near_rim_rows, max_gap=max_gap):
        if shot_like_rim_cluster(
            cluster=cluster,
            ball_df=ball_df,
            hoop_by_frame=hoop_by_frame,
            frame_size=frame_size,
            scale=scale,
            fps=fps,
        ):
            continue

        cluster_indices = {row[0] for row in cluster}
        segment_indices = matching_ball_segment_indices(output_df, cluster_indices)
        remove_indices.update(segment_indices or cluster_indices)

    return remove_indices


def cluster_indexed_frame_rows(
    rows: Sequence[Tuple[int, int, float, float, float]],
    max_gap: int,
) -> List[List[Tuple[int, int, float, float, float]]]:
    clusters: List[List[Tuple[int, int, float, float, float]]] = []
    current: List[Tuple[int, int, float, float, float]] = []
    for row in sorted(rows, key=lambda item: item[1]):
        if current and row[1] - current[-1][1] > max_gap:
            clusters.append(current)
            current = []
        current.append(row)
    if current:
        clusters.append(current)
    return clusters


def shot_like_rim_cluster(
    cluster: Sequence[Tuple[int, int, float, float, float]],
    ball_df: pd.DataFrame,
    hoop_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
    scale: float,
    fps: float,
) -> bool:
    """Return whether a rim-near ball cluster has a plausible shot approach."""

    if not cluster:
        return False

    fps_value = float(fps) if fps and fps > 0 else 30.0
    start_frame = min(row[1] for row in cluster)
    end_frame = max(row[1] for row in cluster)
    min_rim_distance = min(row[4] for row in cluster)
    lookback = max(18, int(round(fps_value * 2.2)))

    prior_df = ball_df[
        (ball_df["frame"] < start_frame)
        & (ball_df["frame"] >= start_frame - lookback)
    ].sort_values("frame")
    if len(prior_df) < 3:
        return False

    prior_distances: List[Tuple[int, float, float, float]] = []
    for _, prior_row in prior_df.iterrows():
        record = prior_row.to_dict()
        try:
            frame = int(record["frame"])
        except (TypeError, ValueError):
            continue
        hoop = nearest_hoop_record(frame, record, hoop_by_frame, search_window=5)
        if hoop is None:
            continue
        distance = math.hypot(
            float(record["x_center"]) - float(hoop["x_center"]),
            float(record["y_center"]) - float(hoop["y_center"]),
        )
        prior_distances.append(
            (
                frame,
                distance,
                float(record["x_center"]),
                float(record["y_center"]),
            )
        )

    if len(prior_distances) < 3:
        return False

    far_prior = [item for item in prior_distances if item[1] >= min_rim_distance + 55.0 * scale]
    if not far_prior:
        return False

    first_far = far_prior[0]
    last_prior = prior_distances[-1]
    approach_drop = first_far[1] - min_rim_distance
    last_drop = last_prior[1] - min_rim_distance
    frame_gap = max(1, start_frame - first_far[0])
    approach_speed = approach_drop / float(frame_gap)
    path_motion = math.hypot(cluster[0][2] - first_far[2], cluster[0][3] - first_far[3])
    cluster_duration = max(1, end_frame - start_frame + 1)

    # A real shot should travel toward the rim from outside the rim zone. A rim
    # false positive usually starts on the rim, jitters there, or gets extended
    # by interpolation without a real approach.
    if approach_drop < 65.0 * scale:
        return False
    if last_drop < 24.0 * scale:
        return False
    if approach_speed < 1.35 * scale and path_motion < 85.0 * scale:
        return False
    if cluster_duration > int(round(fps_value * 1.4)) and approach_speed < 2.0 * scale:
        return False
    return True


def matching_ball_segment_indices(
    output_df: pd.DataFrame,
    cluster_indices: Set[int],
) -> Set[int]:
    """Return all rows in the same ball segments as a suspicious cluster."""

    if "ball_segment" not in output_df.columns or not cluster_indices:
        return set()

    segments: Set[int] = set()
    for index in cluster_indices:
        if index not in output_df.index:
            continue
        value = output_df.at[index, "ball_segment"]
        number = _as_finite_float(value)
        if number is not None:
            segments.add(int(number))
    if not segments:
        return set()

    segment_values = pd.to_numeric(output_df["ball_segment"], errors="coerce")
    mask = output_df["class"].eq("ball") & segment_values.isin(segments)
    return {int(index) for index in output_df.index[mask]}


def static_rim_run_indices(
    rows: Sequence[Tuple[int, int, float, float, float]],
    min_rows: int,
    max_spread: float,
) -> Set[int]:
    if len(rows) < min_rows:
        return set()

    xs = [row[2] for row in rows]
    ys = [row[3] for row in rows]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    median_distance = sorted(row[4] for row in rows)[len(rows) // 2]
    if spread <= max_spread and median_distance <= max_spread * 1.8:
        return {row[0] for row in rows}
    return set()


def edge_static_ball_candidate(
    ball_record: Mapping[str, Any],
    frame_size: Tuple[int, int],
    scale: float,
) -> bool:
    """Return whether a ball is close enough to the frame edge to audit."""

    width, height = frame_size
    ball_x = _as_finite_float(ball_record.get("x_center"))
    ball_y = _as_finite_float(ball_record.get("y_center"))
    if ball_x is None or ball_y is None:
        return False
    margin = max(36.0 * scale, _ball_candidate_size(ball_record) * 1.20)
    return (
        ball_x <= margin
        or ball_x >= float(width) - margin
        or ball_y <= margin
        or ball_y >= float(height) - margin
    )


def static_edge_false_ball_indices(
    near_edge_rows: Sequence[Tuple[int, int, float, float, float]],
    scale: float,
    fps: float,
) -> Set[int]:
    """Remove low-confidence edge candidates that stay almost fixed."""

    if not near_edge_rows:
        return set()

    max_gap = max(2, int(round((float(fps) if fps and fps > 0 else 30.0) * 0.20)))
    min_rows = 4
    max_spread = 18.0 * scale
    remove_indices: Set[int] = set()
    current: List[Tuple[int, int, float, float, float]] = []

    for item in sorted(near_edge_rows, key=lambda row: row[1]):
        if current and item[1] - current[-1][1] > max_gap:
            remove_indices.update(static_edge_run_indices(current, min_rows, max_spread))
            current = []
        current.append(item)

    remove_indices.update(static_edge_run_indices(current, min_rows, max_spread))
    return remove_indices


def static_edge_run_indices(
    rows: Sequence[Tuple[int, int, float, float, float]],
    min_rows: int,
    max_spread: float,
) -> Set[int]:
    if len(rows) < min_rows:
        return set()
    xs = [row[2] for row in rows]
    ys = [row[3] for row in rows]
    confidences = [row[4] for row in rows if math.isfinite(row[4])]
    x_spread = max(xs) - min(xs)
    spread = math.hypot(x_spread, max(ys) - min(ys))
    if not confidences:
        return set()
    avg_confidence = sum(confidences) / len(confidences)
    median_confidence = sorted(confidences)[len(confidences) // 2]
    if spread <= max_spread and (avg_confidence <= 0.24 or median_confidence <= 0.08):
        return {row[0] for row in rows}
    if len(rows) >= 6 and x_spread <= max_spread and (avg_confidence <= 0.20 or median_confidence <= 0.06):
        return {row[0] for row in rows}
    y_spread = max(ys) - min(ys)
    if len(rows) >= 5 and y_spread <= max_spread * 1.65 and median_confidence <= 0.10:
        return {row[0] for row in rows}
    return set()


def static_contextless_ball_indices(
    output_df: pd.DataFrame,
    player_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
    scale: float,
    fps: float,
) -> Set[int]:
    """Remove long, static ball runs that are not near any player."""

    ball_df = output_df[output_df["class"].eq("ball")].copy()
    if ball_df.empty:
        return set()

    fps_value = float(fps) if fps and fps > 0 else 30.0
    max_gap = max(2, int(round(fps_value * 0.20)))
    min_rows = max(7, int(round(fps_value * 0.28)))
    max_spread = 24.0 * scale

    rows: List[Tuple[int, int, float, float, float]] = []
    for index, row in ball_df.dropna(subset=["frame", "x_center", "y_center"]).iterrows():
        rows.append(
            (
                int(index),
                int(row["frame"]),
                float(row["x_center"]),
                float(row["y_center"]),
                float(row.get("confidence", 0.0) or 0.0),
            )
        )

    remove_indices: Set[int] = set()
    current: List[Tuple[int, int, float, float, float]] = []
    for row in sorted(rows, key=lambda item: item[1]):
        if current and row[1] - current[-1][1] > max_gap:
            remove_indices.update(
                static_contextless_run_indices(
                    output_df,
                    current,
                    player_by_frame,
                    frame_size,
                    min_rows,
                    max_spread,
                    scale,
                )
            )
            current = []
        current.append(row)

    remove_indices.update(
        static_contextless_run_indices(
            output_df,
            current,
            player_by_frame,
            frame_size,
            min_rows,
            max_spread,
            scale,
        )
    )
    return remove_indices


def static_contextless_run_indices(
    output_df: pd.DataFrame,
    rows: Sequence[Tuple[int, int, float, float, float]],
    player_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
    min_rows: int,
    max_spread: float,
    scale: float,
) -> Set[int]:
    if len(rows) < min_rows:
        return set()

    xs = [row[2] for row in rows]
    ys = [row[3] for row in rows]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    spread = math.hypot(x_spread, y_spread)
    if spread > max_spread:
        return set()

    near_player_count = 0
    for _, frame, x, y, _ in rows:
        if ball_point_near_player((x, y), frame, player_by_frame, frame_size, scale):
            near_player_count += 1
    near_player_ratio = near_player_count / float(len(rows))
    if near_player_ratio >= 0.30:
        return set()

    cluster_indices = {row[0] for row in rows}
    segment_indices = matching_ball_segment_indices(output_df, cluster_indices)
    return segment_indices or cluster_indices


def ball_point_near_player(
    point: Tuple[float, float],
    frame: int,
    player_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_size: Tuple[int, int],
    scale: float,
) -> bool:
    del frame_size
    margin = 95.0 * scale
    for current_frame in range(frame - 2, frame + 3):
        for player in player_by_frame.get(current_frame, []):
            if not _record_has_bbox(player):
                continue
            if _point_inside_expanded_bbox(point[0], point[1], player, margin):
                return True
    return False


def stabilize_hoop_detections(
    df: pd.DataFrame,
    total_frames: Optional[int],
    frame_size: Tuple[int, int],
    fps: float,
) -> pd.DataFrame:
    """프레임별 링 bbox를 정리하고 짧은 누락 구간을 보간한다.

    링은 선수처럼 빠르게 움직이는 객체가 아니라 카메라 움직임에 따라 천천히 이동하는 context 객체다.
    따라서 짧은 프레임에서 링 탐지가 빠지면 이전/다음 링 위치를 선형 보간해 안정적인 rim zone을 만든다.
    긴 결측이나 큰 화면 전환은 잘못 이어 붙이지 않도록 거리 gate를 적용한다.
    """

    if df.empty or "class" not in df.columns:
        return df

    hoop_df = df[df["class"].eq("hoop")].copy()
    if hoop_df.empty:
        return df

    non_hoop_df = df[~df["class"].eq("hoop")].copy()
    selected = select_primary_hoop_observations(hoop_df, frame_size)
    if selected.empty:
        return non_hoop_df.reset_index(drop=True)

    scale = max(0.85, min(1.45, max(frame_size) / 1280.0))
    fps_value = float(fps) if fps and fps > 0 else 30.0
    max_gap = 28 if fps_value >= 45.0 else 18
    max_gap_distance = 130.0 * scale

    selected_rows = selected.sort_values("frame").to_dict("records")
    stabilized_rows: List[Dict[str, Any]] = []
    for index, left in enumerate(selected_rows):
        stabilized_rows.append(dict(left))
        if index >= len(selected_rows) - 1:
            continue

        right = selected_rows[index + 1]
        frame_gap = int(right["frame"]) - int(left["frame"])
        if frame_gap <= 1 or frame_gap > max_gap:
            continue

        distance = math.hypot(
            float(right["x_center"]) - float(left["x_center"]),
            float(right["y_center"]) - float(left["y_center"]),
        )
        # 카메라 pan/zoom으로 링 위치가 서서히 이동하는 경우만 보간한다.
        # 컷 전환처럼 위치가 크게 바뀌면 다른 장면으로 보고 연결하지 않는다.
        if distance > max(max_gap_distance, 9.0 * scale * frame_gap):
            continue

        for frame in range(int(left["frame"]) + 1, int(right["frame"])):
            ratio = (frame - int(left["frame"])) / float(frame_gap)
            stabilized_rows.append(make_interpolated_hoop_row(left, right, frame, ratio))

    stabilized_hoop_df = pd.DataFrame(stabilized_rows)
    if total_frames is not None and not stabilized_hoop_df.empty:
        frames = pd.to_numeric(stabilized_hoop_df["frame"], errors="coerce")
        stabilized_hoop_df = stabilized_hoop_df[frames.between(0, int(total_frames) - 1)]

    output_df = pd.concat([non_hoop_df, stabilized_hoop_df], ignore_index=True, sort=False)
    return output_df.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)


def select_primary_hoop_observations(
    hoop_df: pd.DataFrame,
    frame_size: Tuple[int, int],
) -> pd.DataFrame:
    """프레임마다 가장 신뢰도 높은 링 bbox 1개를 선택한다."""

    if hoop_df.empty:
        return hoop_df

    numeric_columns = ["frame", "confidence", "x_center", "y_center", "x1", "y1", "x2", "y2"]
    output_df = hoop_df.copy()
    for column in numeric_columns:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")
    output_df = output_df.dropna(subset=["frame", "x_center", "y_center", "x1", "y1", "x2", "y2"])
    if output_df.empty:
        return output_df

    selected_rows: List[Dict[str, Any]] = []
    for _, frame_df in output_df.groupby("frame", sort=True):
        candidates = filter_hoop_candidates(frame_df.to_dict("records"), frame_size)
        if candidates:
            row = dict(candidates[0])
            # 링은 선수처럼 ID별 행동을 추적하는 객체가 아니라 슛 판정의 기준점(context)이다.
            # ByteTrack이 프레임 사이에서 링 ID를 바꿔도 분석에는 같은 링으로 취급해야 하므로 stable id 0으로 통일한다.
            row["track_id"] = 0
            row["hoop_status"] = row.get("hoop_status", "Detected")
            selected_rows.append(row)

    if not selected_rows:
        return output_df.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).sort_values("frame").reset_index(drop=True)


def make_interpolated_hoop_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    frame: int,
    ratio: float,
) -> Dict[str, Any]:
    """두 링 bbox 사이의 짧은 결측 프레임을 선형 보간한다."""

    row = dict(left)
    row["frame"] = int(frame)
    row["class"] = "hoop"
    try:
        left_track_id = int(float(left.get("track_id", -1)))
    except (TypeError, ValueError):
        left_track_id = 0
    row["track_id"] = left_track_id if left_track_id >= 0 else 0
    for column in ["x_center", "y_center", "x1", "y1", "x2", "y2"]:
        row[column] = round(
            float(left[column]) + (float(right[column]) - float(left[column])) * float(ratio),
            2,
        )
    row["confidence"] = round(
        max(0.03, min(float(left.get("confidence", 0.0) or 0.0), float(right.get("confidence", 0.0) or 0.0)) * 0.85),
        3,
    )
    row["hoop_status"] = "Interpolated"
    return row


def draw_detection_boxes(frame: Any, records: Sequence[Mapping[str, Any]]) -> None:
    """한 프레임에 bbox, track_id, class, confidence를 그립니다."""

    for record in records:
        class_name = str(record["class"])
        team_id = normalize_team_id(record.get("team_id"))
        color = TEAM_COLORS.get(team_id, CLASS_COLORS.get(class_name, (255, 255, 255)))
        x1, y1, x2, y2 = _record_bbox(record)
        track_id = int(record.get("track_id", -1))
        confidence = float(record.get("confidence", 0.0))

        thickness = 3 if class_name == "ball" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if class_name == "player" and team_id:
            label = f"{class_name} T{team_id} ID:{track_id} {confidence:.2f}"
        else:
            label = f"{class_name} ID:{track_id} {confidence:.2f}"
        draw_label(frame, label, (x1, y1), color)

        # 공은 bbox가 작아서 잘 안 보이므로 원형 표시를 추가합니다.
        if class_name == "ball":
            center = (int(float(record["x_center"])), int(float(record["y_center"])))
            cv2.circle(frame, center, 8, color, 2)


def render_bbox_video(
    video_path: str,
    output_path: str,
    tracking_df: pd.DataFrame,
    fps: float,
    frame_size: Tuple[int, int],
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> None:
    """최종 후처리 결과를 원본 영상 위에 bbox로 그려 저장합니다."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = create_video_writer(output_path, fps, frame_size)
    records_by_frame = group_records_by_frame(tracking_df)
    frame_id = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        if max_frames is not None and frame_id >= max_frames:
            break

        draw_detection_boxes(frame, records_by_frame.get(frame_id, []))
        writer.write(frame)

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"BBox video rendering: {frame_id} frames", flush=True)

    capture.release()
    writer.release()


def render_ball_trajectory_video(
    video_path: str,
    output_path: str,
    tracking_df: pd.DataFrame,
    fps: float,
    frame_size: Tuple[int, int],
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    tail_length: int = 120,
) -> None:
    """공 위치와 최근 이동 경로를 강조한 검수용 영상을 저장합니다."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = create_video_writer(output_path, fps, frame_size)
    records_by_frame = group_records_by_frame(tracking_df)
    ball_path: List[Tuple[int, int]] = []
    frame_id = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        if max_frames is not None and frame_id >= max_frames:
            break

        records = records_by_frame.get(frame_id, [])
        ball_record = select_ball_record(records)
        if ball_record is not None:
            center = (
                int(round(float(ball_record["x_center"]))),
                int(round(float(ball_record["y_center"]))),
            )
            ball_path.append(center)
            ball_path = ball_path[-max(2, tail_length):]

        draw_trajectory_overlay(frame, ball_path)
        draw_ball_context(frame, records)
        writer.write(frame)

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Trajectory video rendering: {frame_id} frames", flush=True)

    capture.release()
    writer.release()


def select_ball_record(records: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Return the ball row for a frame, preferring the highest confidence row."""

    ball_records = [record for record in records if str(record.get("class")) == "ball"]
    if not ball_records:
        return None
    return max(ball_records, key=lambda row: float(row.get("confidence", 0.0) or 0.0))


def draw_trajectory_overlay(frame: Any, ball_path: Sequence[Tuple[int, int]]) -> None:
    """Draw the recent ball path without running another model pass."""

    if len(ball_path) < 2:
        return

    for index in range(1, len(ball_path)):
        age = len(ball_path) - index
        thickness = 2 if age > 30 else 3
        cv2.line(frame, ball_path[index - 1], ball_path[index], (0, 220, 255), thickness)

    for index, point in enumerate(ball_path[-12:]):
        radius = max(3, 8 - (len(ball_path[-12:]) - index) // 2)
        cv2.circle(frame, point, radius, (0, 220, 255), -1)


def draw_ball_context(frame: Any, records: Sequence[Mapping[str, Any]]) -> None:
    """Draw ball and hoop boxes on the trajectory video for context."""

    for record in records:
        class_name = str(record.get("class"))
        if class_name not in {"ball", "hoop"}:
            continue

        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        x1, y1, x2, y2 = _record_bbox(record)
        thickness = 3 if class_name == "ball" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = class_name if class_name == "hoop" else "ball path"
        draw_label(frame, label, (x1, y1), color)


def group_records_by_frame(df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    """Group final tracking rows by frame number."""

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    if df.empty:
        return grouped

    for record in df.to_dict("records"):
        frame = int(record["frame"])
        grouped.setdefault(frame, []).append(record)
    return grouped


def draw_label(
    frame: Any,
    label: str,
    top_left: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """Draw a readable label background and text above a bbox."""

    x, y = top_left
    y = max(18, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (x, y - text_height - 8),
        (x + text_width + 6, y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 3, y - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _record_bbox(record: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    """Convert a record bbox to int coordinates."""

    return (
        int(float(record["x1"])),
        int(float(record["y1"])),
        int(float(record["x2"])),
        int(float(record["y2"])),
    )


def normalize_team_id(value: Any) -> str:
    """Return a stable string team id for labels and colors."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _validate_input_file(path: str, label: str) -> None:
    """Check whether a required input file exists."""

    if not Path(path).exists():
        raise FileNotFoundError(f"{label} file not found: {path}")


__all__ = ["run_yolo_tracking_pipeline"]
