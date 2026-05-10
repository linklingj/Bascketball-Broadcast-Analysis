"""공 좌표 정제 및 검증 영상 생성 모듈.

이 파일은 슛/패스/리바운드 같은 통계 계산을 하지 않습니다.
목표는 통계 계산 전에 사용할 수 있는 "깨끗한 공 좌표"를 만들고, 그 결과를 눈으로 검증할 수
있도록 별도 CSV와 오버레이 영상을 생성하는 것입니다.

생성 파일:
- raw_detection_verify.csv: YOLO가 프레임별로 탐지한 원본 공 후보 좌표
- cleaned_tracking_results.csv: 이상치 제거, 짧은 결측 보간, moving average smoothing을 거친 결과
- output_verification.mp4: 원본 공 후보(빨간색)와 정제 공 좌표(초록색)를 함께 그린 검증 영상
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


BALL_CLASS_NAMES = {"ball", "sports ball", "basketball", "frisbee"}
BASE_TRACKING_COLUMNS = [
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


@dataclass
class BallRefinerConfig:
    """공 좌표 정제에 필요한 튜닝 파라미터입니다.

    팀원이 가장 자주 조정할 값:
    - max_step_px_per_frame: 한 프레임 사이에 공이 이동할 수 있는 최대 픽셀 거리
    - max_interpolation_gap: 선형 보간으로 메울 최대 결측 프레임 수
    - smoothing_window: moving average에 사용할 프레임 수
    """

    # 물리적 속도 한계치 필터링:
    # 1080p/30fps 방송 영상 기준으로 160px/frame을 넘는 이동은 대부분 false positive입니다.
    # 카메라 줌/해상도가 다르면 frame_size에 맞춰 내부에서 조금 보정합니다.
    max_step_px_per_frame: float = 160.0
    # 정제 후에도 이 값을 넘는 연속 프레임 이동이 남으면 통계 계산에서 위험합니다.
    # 720p/1080p 영상 모두에서 너무 공격적으로 지우지 않도록 해상도 보정을 적용합니다.
    strict_max_step_px_per_frame: float = 135.0
    # 탐지를 놓친 구간이 너무 길면 보간이 오히려 거짓 좌표를 만들 수 있으므로 5프레임까지만 채웁니다.
    max_interpolation_gap: int = 5
    # 선수 가림 등으로 짧게 놓친 직후, 이전 속도 벡터로 예측할 최대 프레임 수입니다.
    max_prediction_gap: int = 3
    # 1~2px 수준의 흔들림을 줄이기 위한 이동 평균 창입니다. 홀수 3 또는 5를 권장합니다.
    smoothing_window: int = 3
    # 새 segment를 시작할 때 너무 낮은 confidence 후보를 무조건 믿지 않기 위한 기준입니다.
    restart_confidence_threshold: float = 0.25
    # 긴 공백 뒤 완전히 다른 위치에서 다시 시작하는 후보는 confidence가 충분히 높아야 인정합니다.
    high_confidence_restart_threshold: float = 0.65
    # 화면 가장자리 후보는 관중석/광고판/조명 오탐이 많아 별도 감점합니다.
    edge_margin_px: float = 24.0
    edge_confidence_threshold: float = 0.55
    # 선수/심판/골대 같은 농구 맥락에서 너무 멀리 떨어진 낮은 confidence 후보는 제거합니다.
    context_distance_px: float = 420.0
    context_confidence_threshold: float = 0.18
    # 시각화에서 최근 몇 프레임의 공 궤적을 tail로 보여줄지 결정합니다.
    tail_length: int = 10
    # 정제 공 bbox를 새로 만들 때 사용할 기본 크기입니다.
    default_ball_box_size: float = 18.0


def build_adaptive_refiner_config(
    frame_size: Tuple[int, int],
    fps: float,
) -> BallRefinerConfig:
    """영상 해상도/FPS에 맞춰 공 좌표 정제 임계값을 자동으로 만든다.

    특정 4개 영상에만 맞춘 값이 아니라, 입력 영상의 기본 메타데이터를 읽어
    프레임당 이동 가능 거리, context 거리, 기본 공 bbox 크기를 자동 보정한다.
    이렇게 해야 다른 경기 영상을 넣어도 비슷한 기준으로 false positive와 결측 보간을 처리할 수 있다.
    """

    width, height = frame_size
    max_dim = max(float(width), float(height), 1.0)
    fps_value = float(fps) if fps and fps > 0 else 30.0

    # FPS가 높을수록 같은 실제 속도라도 한 프레임에서 이동하는 픽셀 거리는 작아진다.
    # 단, 너무 과하게 조정하면 슛/패스처럼 빠른 움직임을 제거할 수 있어 보수적인 범위로 제한한다.
    fps_factor = max(0.75, min(1.25, 30.0 / fps_value))

    # 해상도가 커지면 선수/골대 주변 context 거리도 픽셀 단위로 함께 커진다.
    resolution_factor = max(0.85, min(1.35, max_dim / 1280.0))

    return BallRefinerConfig(
        max_step_px_per_frame=165.0 * fps_factor,
        strict_max_step_px_per_frame=138.0 * fps_factor,
        max_interpolation_gap=5,
        max_prediction_gap=5 if fps_value >= 45.0 else 3,
        smoothing_window=3,
        restart_confidence_threshold=0.24,
        high_confidence_restart_threshold=0.62,
        edge_margin_px=max(20.0, min(42.0, max_dim * 0.018)),
        edge_confidence_threshold=0.55,
        context_distance_px=420.0 * resolution_factor,
        context_confidence_threshold=0.18,
        tail_length=10,
        default_ball_box_size=max(12.0, min(26.0, max_dim * 0.012)),
    )


class BallCoordinateRefiner:
    """공 좌표 검수 CSV, 정제 CSV, 검증 영상을 생성하는 클래스입니다."""

    def __init__(self, config: Optional[BallRefinerConfig] = None) -> None:
        self.config = config or BallRefinerConfig()

    def export_raw_detection_verify(self, raw_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
        """원본 공 후보 좌표를 검수용 CSV로 저장합니다.

        Input:
            raw_df: YOLO가 만든 원본 detection DataFrame. 프레임당 공 후보가 여러 개일 수 있습니다.
            output_path: raw_detection_verify.csv 저장 경로

        Output:
            저장된 원본 공 후보 DataFrame
        """

        raw_ball_df = self._ball_rows(raw_df).copy()
        if raw_ball_df.empty:
            verify_df = pd.DataFrame(
                columns=[
                    "frame",
                    "raw_candidate_index",
                    "track_id",
                    "confidence",
                    "x_center",
                    "y_center",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "bbox_width",
                    "bbox_height",
                    "bbox_area",
                ]
            )
        else:
            raw_ball_df = self._ensure_numeric(raw_ball_df, ["frame", "confidence", "x_center", "y_center", "x1", "y1", "x2", "y2"])
            raw_ball_df = raw_ball_df.sort_values(["frame", "confidence"], ascending=[True, False]).copy()
            raw_ball_df["raw_candidate_index"] = raw_ball_df.groupby("frame").cumcount()
            raw_ball_df["bbox_width"] = (raw_ball_df["x2"] - raw_ball_df["x1"]).abs()
            raw_ball_df["bbox_height"] = (raw_ball_df["y2"] - raw_ball_df["y1"]).abs()
            raw_ball_df["bbox_area"] = raw_ball_df["bbox_width"] * raw_ball_df["bbox_height"]
            verify_df = raw_ball_df[
                [
                    "frame",
                    "raw_candidate_index",
                    "track_id",
                    "confidence",
                    "x_center",
                    "y_center",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "bbox_width",
                    "bbox_height",
                    "bbox_area",
                ]
            ].copy()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        verify_df.to_csv(output, index=False, encoding="utf-8-sig")
        return verify_df

    def create_cleaned_tracking_results(
        self,
        tracking_df: pd.DataFrame,
        frame_size: Tuple[int, int],
        total_frames: Optional[int],
        output_path: str,
    ) -> pd.DataFrame:
        """공 좌표를 정제한 cleaned_tracking_results.csv를 생성합니다.

        처리 순서:
        1. 프레임별 공 후보를 하나로 정리합니다.
        2. 물리적으로 불가능한 큰 이동을 outlier로 제거합니다.
        3. 최대 5프레임 이내 결측 구간은 선형 보간합니다.
        4. 짧은 occlusion 구간은 이전 속도 벡터로 예측합니다.
        5. Moving Average로 미세한 jitter를 줄입니다.
        """

        output_df = tracking_df.copy()
        if output_df.empty:
            self._write_csv(output_df, output_path)
            return output_df

        non_ball_df = output_df[~output_df.get("class", pd.Series(dtype=str)).isin(BALL_CLASS_NAMES)].copy()
        ball_df = self._select_frame_ball_observations(output_df)
        context_by_frame = self._context_points_by_frame(non_ball_df)

        if ball_df.empty:
            cleaned_df = non_ball_df
        else:
            accepted_points = self._remove_motion_outliers(ball_df, frame_size, context_by_frame)
            timeline = self._build_clean_ball_timeline(accepted_points, total_frames=total_frames)
            cleaned_ball_df = self._smooth_ball_timeline(timeline, frame_size)
            cleaned_df = pd.concat([non_ball_df, cleaned_ball_df], ignore_index=True, sort=False)

        cleaned_df = self._normalize_output_columns(cleaned_df)
        self._write_csv(cleaned_df, output_path)
        return cleaned_df

    def render_verification_video(
        self,
        video_path: str,
        raw_verify_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> None:
        """원본 공 후보와 정제 공 좌표를 함께 표시한 검증 영상을 생성합니다.

        시각화 규칙:
        - 빨간색 점: 원본 YOLO 공 후보
        - 초록색 점: 정제된 최종 공 좌표
        - 초록색 선: 최근 tail_length 프레임의 정제 공 이동 경로
        - 좌상단 텍스트: 현재 프레임 번호와 공 좌표 상태
        """

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        raw_by_frame = self._records_by_frame(raw_verify_df)
        cleaned_ball_df = self._ball_rows(cleaned_df)
        cleaned_by_frame = self._records_by_frame(cleaned_ball_df)
        tail: List[Tuple[int, int]] = []
        frame_id = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            if max_frames is not None and frame_id >= max_frames:
                break

            raw_records = raw_by_frame.get(frame_id, [])
            for record in raw_records:
                point = self._center_point(record)
                if point is not None:
                    # 원본 후보는 빨간색 작은 점으로 표시합니다. 한 프레임에 여러 후보가 있을 수 있습니다.
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)

            cleaned_records = cleaned_by_frame.get(frame_id, [])
            status = "Missing"
            if cleaned_records:
                record = cleaned_records[0]
                point = self._center_point(record)
                status = str(record.get("ball_status", "Detected"))
                if point is not None:
                    # 정제 좌표는 초록색 큰 점으로 표시합니다. 이후 통계 계산에는 이 좌표를 사용합니다.
                    reliable = str(record.get("is_reliable", True)).strip().lower() not in {"false", "0", "no"}
                    color = (0, 220, 0) if reliable else (0, 220, 255)
                    # 초록색은 신뢰 좌표, 노란색은 주의가 필요한 좌표입니다.
                    cv2.circle(frame, point, 8, color, -1)
                    if reliable:
                        tail.append(point)
                        tail = tail[-self.config.tail_length :]
                    status = f"{status}/{'Reliable' if reliable else 'Check'}"

            if len(tail) >= 2:
                for left, right in zip(tail, tail[1:]):
                    cv2.line(frame, left, right, (0, 220, 0), 2)

            self._draw_status_box(frame, frame_id, status)
            writer.write(frame)

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"Verification video rendering: {frame_id} frames", flush=True)

        capture.release()
        writer.release()

    def _select_frame_ball_observations(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """프레임별 공 row가 여러 개일 때 confidence가 가장 높은 후보를 선택합니다."""

        ball_df = self._ball_rows(tracking_df).copy()
        if ball_df.empty:
            return ball_df

        ball_df = self._ensure_numeric(ball_df, ["frame", "confidence", "x_center", "y_center", "x1", "y1", "x2", "y2"])
        ball_df = ball_df.dropna(subset=["frame", "x_center", "y_center"]).copy()
        if ball_df.empty:
            return ball_df

        ball_df["bbox_size"] = self._bbox_size_series(ball_df)
        ball_df = ball_df.sort_values(["frame", "confidence"], ascending=[True, False])
        return ball_df.drop_duplicates("frame", keep="first").sort_values("frame").reset_index(drop=True)

    def _remove_motion_outliers(
        self,
        ball_df: pd.DataFrame,
        frame_size: Tuple[int, int],
        context_by_frame: Mapping[int, Sequence[Tuple[float, float]]],
    ) -> List[Dict[str, Any]]:
        """물리적 속도 한계치를 넘는 공 좌표를 outlier로 제거합니다.

        수학적 근거:
        distance = sqrt((x_t - x_prev)^2 + (y_t - y_prev)^2)
        distance가 허용 이동량(max_step_px_per_frame * frame_gap)을 초과하면 같은 공의 이동으로 보기 어렵습니다.
        """

        max_step = self._scaled_max_step(frame_size)
        accepted: List[Dict[str, Any]] = []
        segment_id = 0

        for row in ball_df.to_dict("records"):
            point = dict(row)
            point["ball_status"] = "Detected"
            point["cleaning_note"] = "accepted"
            point["raw_x_center"] = point.get("x_center")
            point["raw_y_center"] = point.get("y_center")
            point["is_reliable"] = True
            point["reliability_reason"] = "accepted"

            confidence = float(point.get("confidence", 0.0) or 0.0)
            near_edge = self._is_near_frame_edge(point, frame_size)
            context_distance = self._nearest_context_distance(point, context_by_frame)
            point["context_distance"] = round(context_distance, 2) if np.isfinite(context_distance) else None

            # 화면 가장자리 후보는 광고판/관중석/조명 오탐이 섞이기 쉽습니다.
            # 첫 좌표부터 가장자리 저신뢰 후보이면 추적 시작점으로 사용하지 않습니다.
            if not accepted and near_edge and confidence < self.config.edge_confidence_threshold:
                point["cleaning_note"] = "removed_edge_low_confidence"
                continue

            # 선수/심판/골대와 너무 멀리 떨어진 낮은 confidence 후보는 공일 가능성이 낮습니다.
            if (
                np.isfinite(context_distance)
                and context_distance > self.config.context_distance_px
                and confidence < self.config.context_confidence_threshold
            ):
                point["cleaning_note"] = "removed_far_from_context"
                continue

            if not accepted:
                point["ball_segment"] = segment_id
                accepted.append(point)
                continue

            previous = accepted[-1]
            frame_gap = int(point["frame"]) - int(previous["frame"])
            if frame_gap <= 0:
                continue

            distance = float(np.hypot(
                float(point["x_center"]) - float(previous["x_center"]),
                float(point["y_center"]) - float(previous["y_center"]),
            ))
            # 결측 구간이 길수록 허용 거리를 어느 정도 늘리되, 무한정 키우지는 않습니다.
            # 허용 거리가 너무 커지면 화면 반대편 false positive도 새 공으로 받아들이게 됩니다.
            allowed_distance = max_step * max(1, min(frame_gap, 3))

            # 바로 다음 프레임에서 허용치를 넘는 큰 점프는 false positive일 가능성이 높으므로 제거합니다.
            if frame_gap == 1 and distance > max_step:
                point["cleaning_note"] = "removed_impossible_speed"
                continue

            # 5프레임 이내처럼 짧은 구간에서 큰 점프가 나오면 새 공이라기보다 오탐일 가능성이 큽니다.
            # 이 구간은 뒤에서 선형 보간할 수 있으므로 과감히 제거합니다.
            if frame_gap <= self.config.max_interpolation_gap + 1 and distance > allowed_distance:
                point["cleaning_note"] = "removed_short_gap_jump"
                continue

            # 여러 프레임을 놓친 뒤 아주 멀리 떨어진 낮은 confidence 후보가 나오면 노이즈로 간주합니다.
            if distance > allowed_distance and confidence < self.config.restart_confidence_threshold:
                point["cleaning_note"] = "removed_far_low_confidence"
                continue

            # 긴 공백 뒤 화면 가장자리에서 다시 잡힌 후보는 새 segment로도 인정하지 않습니다.
            if distance > allowed_distance and near_edge:
                point["cleaning_note"] = "removed_edge_restart"
                continue

            if (
                distance > allowed_distance
                and np.isfinite(context_distance)
                and context_distance > self.config.context_distance_px
                and confidence < self.config.high_confidence_restart_threshold
            ):
                point["cleaning_note"] = "removed_far_restart_without_context"
                continue

            # 긴 공백 뒤 높은 confidence 후보는 새 segment로 인정합니다.
            # segment가 다르면 보간 단계에서 이전 좌표와 억지로 이어 붙이지 않습니다.
            if distance > allowed_distance:
                if confidence < self.config.high_confidence_restart_threshold:
                    point["cleaning_note"] = "removed_weak_new_segment"
                    continue
                segment_id += 1
                point["cleaning_note"] = "accepted_new_segment"

            point["ball_segment"] = segment_id
            point["motion_distance"] = round(distance, 3)
            point["frame_gap_from_previous_ball"] = frame_gap
            accepted.append(point)

        return self._prune_weak_segments(accepted)

    def _build_clean_ball_timeline(
        self,
        accepted_points: Sequence[Mapping[str, Any]],
        total_frames: Optional[int],
    ) -> pd.DataFrame:
        """accepted 공 좌표 사이의 짧은 결측을 보간하고 occlusion 구간을 예측합니다."""

        if not accepted_points:
            return pd.DataFrame(columns=BASE_TRACKING_COLUMNS + ["ball_status", "cleaning_note"])

        sorted_points = sorted(accepted_points, key=lambda item: int(item["frame"]))
        rows: List[Dict[str, Any]] = []

        for index, point in enumerate(sorted_points):
            rows.append(self._make_ball_row(point, status="Detected", note=str(point.get("cleaning_note", "accepted"))))

            if index >= len(sorted_points) - 1:
                continue

            next_point = sorted_points[index + 1]
            frame_gap = int(next_point["frame"]) - int(point["frame"])
            if frame_gap <= 1:
                continue

            same_segment = int(point.get("ball_segment", -1)) == int(next_point.get("ball_segment", -2))
            missing_count = frame_gap - 1

            if same_segment and missing_count <= self.config.max_interpolation_gap:
                # 결측치 보간 로직:
                # x/y를 좌우 anchor 사이에서 선형 보간합니다.
                # x_t = x0 + (x1 - x0) * ratio, y_t = y0 + (y1 - y0) * ratio
                rows.extend(self._interpolate_between(point, next_point, missing_count))
            elif same_segment:
                # 긴 결측 구간은 전체를 보간하지 않고, 직후 몇 프레임만 이전 속도 벡터로 예측합니다.
                rows.extend(self._predict_after(point, sorted_points, index, missing_count))

        if total_frames is not None and sorted_points:
            last_point = sorted_points[-1]
            remaining = int(total_frames) - int(last_point["frame"]) - 1
            if remaining > 0:
                rows.extend(self._predict_after(last_point, sorted_points, len(sorted_points) - 1, remaining))

        return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)

    def _prune_weak_segments(self, points: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """짧고 약한 공 segment를 제거합니다.

        긴 공백 뒤에 1~2프레임만 나타나는 공 후보는 실제 공보다 오탐일 가능성이 높습니다.
        다만 confidence가 높고 선수/골대 근처라면 실제 빠른 패스나 슛일 수 있으므로 유지합니다.
        """

        if not points:
            return []

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for point in points:
            grouped.setdefault(int(point.get("ball_segment", 0)), []).append(dict(point))

        kept: List[Dict[str, Any]] = []
        for segment_points in grouped.values():
            confidences = [float(point.get("confidence", 0.0) or 0.0) for point in segment_points]
            context_distances = [
                float(point["context_distance"])
                for point in segment_points
                if point.get("context_distance") is not None
            ]
            max_confidence = max(confidences) if confidences else 0.0
            min_context_distance = min(context_distances) if context_distances else float("inf")

            if (
                len(segment_points) >= 3
                or max_confidence >= self.config.high_confidence_restart_threshold
                or min_context_distance <= self.config.context_distance_px
            ):
                kept.extend(segment_points)

        return sorted(kept, key=lambda item: int(item["frame"]))

    def _interpolate_between(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        missing_count: int,
    ) -> List[Dict[str, Any]]:
        """두 검출점 사이의 짧은 결측 프레임을 선형 보간합니다."""

        rows: List[Dict[str, Any]] = []
        left_frame = int(left["frame"])
        right_frame = int(right["frame"])
        total_gap = float(right_frame - left_frame)
        for frame in range(left_frame + 1, right_frame):
            ratio = float(frame - left_frame) / total_gap
            x = float(left["x_center"]) + (float(right["x_center"]) - float(left["x_center"])) * ratio
            y = float(left["y_center"]) + (float(right["y_center"]) - float(left["y_center"])) * ratio
            confidence = max(0.05, min(float(left.get("confidence", 0.0) or 0.0), float(right.get("confidence", 0.0) or 0.0)) * 0.8)
            rows.append(
                self._make_ball_row(
                    left,
                    frame=frame,
                    x_center=x,
                    y_center=y,
                    confidence=confidence,
                    status="Interpolated",
                    note=f"linear_interpolation_gap_{missing_count}",
                )
            )
        return rows

    def _predict_after(
        self,
        point: Mapping[str, Any],
        sorted_points: Sequence[Mapping[str, Any]],
        point_index: int,
        missing_count: int,
    ) -> List[Dict[str, Any]]:
        """이전 속도 벡터를 이용해 짧은 occlusion 구간을 예측합니다."""

        velocity = self._estimate_velocity(sorted_points, point_index)
        if velocity is None:
            return []

        vx, vy = velocity
        rows: List[Dict[str, Any]] = []
        predict_count = min(self.config.max_prediction_gap, missing_count)
        for step in range(1, predict_count + 1):
            frame = int(point["frame"]) + step
            x = float(point["x_center"]) + vx * step
            y = float(point["y_center"]) + vy * step
            confidence = max(0.01, float(point.get("confidence", 0.0) or 0.0) * (0.75**step))
            rows.append(
                self._make_ball_row(
                    point,
                    frame=frame,
                    x_center=x,
                    y_center=y,
                    confidence=confidence,
                    status="Predicted",
                    note="linear_prediction_occlusion",
                )
            )
        return rows

    def _smooth_ball_timeline(self, timeline: pd.DataFrame, frame_size: Tuple[int, int]) -> pd.DataFrame:
        """Moving Average로 1~2픽셀 단위의 미세한 좌표 떨림을 줄입니다."""

        if timeline.empty:
            return timeline

        smoothed_parts: List[pd.DataFrame] = []
        window = max(1, int(self.config.smoothing_window))
        if window % 2 == 0:
            window += 1

        for _, segment_df in timeline.groupby("ball_segment", sort=True):
            segment_df = segment_df.sort_values("frame").copy()
            # center=True를 사용해 현재 프레임 주변의 좌우 값을 함께 반영합니다.
            # min_periods=1 덕분에 segment 초반/후반 프레임도 제거되지 않습니다.
            segment_df["x_center"] = segment_df["x_center"].rolling(window=window, center=True, min_periods=1).mean()
            segment_df["y_center"] = segment_df["y_center"].rolling(window=window, center=True, min_periods=1).mean()
            segment_df = self._rebuild_ball_bbox(segment_df)
            smoothed_parts.append(segment_df)

        smoothed_df = pd.concat(smoothed_parts, ignore_index=True).sort_values("frame").reset_index(drop=True)
        smoothed_df = self._remove_implausible_clean_jumps(smoothed_df, frame_size)
        return self._append_reliability_columns(smoothed_df, frame_size)

    def _remove_implausible_clean_jumps(self, ball_df: pd.DataFrame, frame_size: Tuple[int, int]) -> pd.DataFrame:
        """smoothing 이후에도 남은 비정상 점프를 한 번 더 제거합니다.

        앞 단계는 원본 detection 기준 필터이고, smoothing 이후 좌표가 조금 변할 수 있습니다.
        따라서 최종 cleaned CSV 저장 직전에 연속 프레임 큰 점프를 다시 검사합니다.
        """

        if ball_df.empty:
            return ball_df

        strict_step = self._scaled_strict_step(frame_size)
        kept_rows: List[Mapping[str, Any]] = []

        for row in ball_df.sort_values("frame").to_dict("records"):
            if not kept_rows:
                kept_rows.append(row)
                continue

            previous = kept_rows[-1]
            frame_gap = int(row["frame"]) - int(previous["frame"])
            if frame_gap <= 0:
                continue

            distance = float(np.hypot(
                float(row["x_center"]) - float(previous["x_center"]),
                float(row["y_center"]) - float(previous["y_center"]),
            ))
            allowed = strict_step * max(1, min(frame_gap, 2))

            # 최종 CSV에서는 짧은 구간의 큰 점프를 남기지 않습니다.
            # 통계 계산에서 틀린 공 좌표 하나가 슛/패스 이벤트를 크게 흔들 수 있기 때문입니다.
            if frame_gap <= self.config.max_interpolation_gap + 1 and distance > allowed:
                row["is_reliable"] = False
                row["reliability_reason"] = "removed_final_jump_gate"
                continue

            kept_rows.append(row)

        if not kept_rows:
            return ball_df.iloc[0:0].copy()
        return pd.DataFrame(kept_rows).sort_values("frame").reset_index(drop=True)

    def _append_reliability_columns(self, ball_df: pd.DataFrame, frame_size: Tuple[int, int]) -> pd.DataFrame:
        """정제 공 좌표에 신뢰도 진단 컬럼을 추가합니다.

        추가 컬럼:
        - is_reliable: 이후 통계 계산에 사용할 수 있는 좌표인지 여부
        - reliability_reason: 신뢰/주의 사유
        - jump_distance: 직전 정제 공 좌표와의 이동 거리
        """

        if ball_df.empty:
            return ball_df

        output_df = ball_df.sort_values("frame").copy()
        strict_step = self._scaled_strict_step(frame_size)
        previous_row: Optional[pd.Series] = None
        jump_distances: List[Optional[float]] = []
        reliable_values: List[bool] = []
        reasons: List[str] = []

        for _, row in output_df.iterrows():
            confidence = float(row.get("confidence", 0.0) or 0.0)
            status = str(row.get("ball_status", "Detected"))
            reason = str(row.get("cleaning_note", "accepted"))
            is_reliable = True
            jump_distance: Optional[float] = None

            if previous_row is not None:
                frame_gap = int(row["frame"]) - int(previous_row["frame"])
                if frame_gap > 0:
                    same_segment = int(row.get("ball_segment", -1)) == int(previous_row.get("ball_segment", -2))
                    continuous_gap = frame_gap <= self.config.max_interpolation_gap + 1
                    # segment가 바뀌었거나 긴 결측 뒤 다시 탐지된 좌표는 "연속 이동"으로 보지 않는다.
                    # 이후 슛/패스 계산에서 두 지점을 직선으로 연결해 속도를 계산하면 큰 오차가 생기므로
                    # jump_distance를 비워 두고 ball_segment로 구간을 분리해서 해석하게 한다.
                    if same_segment and continuous_gap:
                        jump_distance = float(np.hypot(
                            float(row["x_center"]) - float(previous_row["x_center"]),
                            float(row["y_center"]) - float(previous_row["y_center"]),
                        ))
                        allowed = strict_step * max(1, min(frame_gap, 2))
                        if jump_distance > allowed:
                            is_reliable = False
                            reason = "warning_large_clean_jump"

            if status == "Predicted" and confidence < 0.02:
                is_reliable = False
                reason = "warning_low_confidence_prediction"

            if status == "Detected" and confidence < 0.01:
                is_reliable = False
                reason = "warning_very_low_confidence_detection"

            jump_distances.append(round(jump_distance, 3) if jump_distance is not None else None)
            reliable_values.append(is_reliable)
            reasons.append(reason)
            previous_row = row

        output_df["jump_distance"] = jump_distances
        output_df["is_reliable"] = reliable_values
        output_df["reliability_reason"] = reasons
        return output_df

    def _make_ball_row(
        self,
        source: Mapping[str, Any],
        frame: Optional[int] = None,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        confidence: Optional[float] = None,
        status: str = "Detected",
        note: str = "accepted",
    ) -> Dict[str, Any]:
        """정제된 공 row 1개를 표준 tracking_results 형식으로 만듭니다."""

        row = dict(source)
        row["frame"] = int(source["frame"] if frame is None else frame)
        row["track_id"] = 0
        row["class"] = "ball"
        row["confidence"] = round(float(source.get("confidence", 0.0) if confidence is None else confidence), 3)
        row["x_center"] = round(float(source["x_center"] if x_center is None else x_center), 2)
        row["y_center"] = round(float(source["y_center"] if y_center is None else y_center), 2)
        row["ball_status"] = status
        row["cleaning_note"] = note
        row["raw_x_center"] = source.get("raw_x_center", source.get("x_center"))
        row["raw_y_center"] = source.get("raw_y_center", source.get("y_center"))
        row["ball_segment"] = int(source.get("ball_segment", 0))
        row["bbox_size"] = float(source.get("bbox_size", self.config.default_ball_box_size) or self.config.default_ball_box_size)
        return row

    def _estimate_velocity(
        self,
        sorted_points: Sequence[Mapping[str, Any]],
        point_index: int,
    ) -> Optional[Tuple[float, float]]:
        """현재 point 직전의 같은 segment 좌표를 찾아 px/frame 속도 벡터를 계산합니다."""

        if point_index <= 0:
            return None
        current = sorted_points[point_index]
        current_segment = int(current.get("ball_segment", -1))
        for prev_index in range(point_index - 1, -1, -1):
            previous = sorted_points[prev_index]
            if int(previous.get("ball_segment", -2)) != current_segment:
                continue
            frame_delta = int(current["frame"]) - int(previous["frame"])
            if frame_delta <= 0:
                continue
            vx = (float(current["x_center"]) - float(previous["x_center"])) / frame_delta
            vy = (float(current["y_center"]) - float(previous["y_center"])) / frame_delta
            return vx, vy
        return None

    def _rebuild_ball_bbox(self, ball_df: pd.DataFrame) -> pd.DataFrame:
        """중심 좌표가 smoothing으로 바뀐 뒤 bbox도 같은 위치로 다시 맞춥니다."""

        output_df = ball_df.copy()
        sizes = pd.to_numeric(output_df.get("bbox_size", self.config.default_ball_box_size), errors="coerce").fillna(self.config.default_ball_box_size)
        half = sizes / 2.0
        output_df["x1"] = (output_df["x_center"] - half).round(2)
        output_df["y1"] = (output_df["y_center"] - half).round(2)
        output_df["x2"] = (output_df["x_center"] + half).round(2)
        output_df["y2"] = (output_df["y_center"] + half).round(2)
        output_df["x_center"] = output_df["x_center"].round(2)
        output_df["y_center"] = output_df["y_center"].round(2)
        return output_df

    def _scaled_max_step(self, frame_size: Tuple[int, int]) -> float:
        """영상 해상도에 따라 속도 threshold를 약간 보정합니다."""

        width, height = frame_size
        scale = max(width, height) / 1920.0
        return float(self.config.max_step_px_per_frame) * max(0.75, scale)

    def _scaled_strict_step(self, frame_size: Tuple[int, int]) -> float:
        """최종 cleaned CSV용 더 보수적인 속도 threshold를 계산합니다."""

        width, height = frame_size
        scale = max(width, height) / 1920.0
        return float(self.config.strict_max_step_px_per_frame) * max(0.75, scale)

    def _is_near_frame_edge(self, point: Mapping[str, Any], frame_size: Tuple[int, int]) -> bool:
        """공 후보가 화면 가장자리 근처인지 판단합니다."""

        width, height = frame_size
        try:
            x = float(point["x_center"])
            y = float(point["y_center"])
        except (KeyError, TypeError, ValueError):
            return True
        margin = float(self.config.edge_margin_px)
        return x <= margin or y <= margin or x >= width - margin or y >= height - margin

    def _context_points_by_frame(self, non_ball_df: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
        """선수/심판/골대 중심점을 프레임별로 묶습니다.

        낮은 confidence 공 후보가 농구 맥락 객체에서 너무 멀리 떨어져 있으면 오탐으로 볼 수 있습니다.
        """

        if non_ball_df.empty:
            return {}

        context_df = non_ball_df[non_ball_df.get("class", pd.Series(dtype=str)).isin({"player", "referee", "hoop"})].copy()
        if context_df.empty:
            return {}

        context_df = self._ensure_numeric(context_df, ["frame", "x_center", "y_center"])
        context_df = context_df.dropna(subset=["frame", "x_center", "y_center"])
        grouped: Dict[int, List[Tuple[float, float]]] = {}
        for row in context_df.to_dict("records"):
            grouped.setdefault(int(row["frame"]), []).append((float(row["x_center"]), float(row["y_center"])))
        return grouped

    def _nearest_context_distance(
        self,
        point: Mapping[str, Any],
        context_by_frame: Mapping[int, Sequence[Tuple[float, float]]],
    ) -> float:
        """공 후보와 같은 프레임 주변의 농구 맥락 객체까지 가장 가까운 거리를 계산합니다."""

        try:
            frame = int(point["frame"])
            x = float(point["x_center"])
            y = float(point["y_center"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

        candidates: List[Tuple[float, float]] = []
        for offset in (-1, 0, 1):
            candidates.extend(context_by_frame.get(frame + offset, []))
        if not candidates:
            return float("inf")
        return min(float(np.hypot(x - cx, y - cy)) for cx, cy in candidates)

    def _bbox_size_series(self, df: pd.DataFrame) -> pd.Series:
        """기존 bbox 크기에서 공 bbox 대표 크기를 계산합니다."""

        if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
            width = (pd.to_numeric(df["x2"], errors="coerce") - pd.to_numeric(df["x1"], errors="coerce")).abs()
            height = (pd.to_numeric(df["y2"], errors="coerce") - pd.to_numeric(df["y1"], errors="coerce")).abs()
            size = pd.concat([width, height], axis=1).max(axis=1)
            return size.fillna(self.config.default_ball_box_size).clip(lower=8.0, upper=40.0)
        return pd.Series(self.config.default_ball_box_size, index=df.index)

    def _normalize_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 tracking_results.csv 컬럼을 앞쪽에 유지하고 정제 메타데이터를 뒤에 붙입니다."""

        if df.empty:
            return df
        for column in BASE_TRACKING_COLUMNS:
            if column not in df.columns:
                df[column] = None
        preferred = BASE_TRACKING_COLUMNS + [
            "ball_status",
            "cleaning_note",
            "raw_x_center",
            "raw_y_center",
            "ball_segment",
            "is_reliable",
            "reliability_reason",
            "jump_distance",
            "context_distance",
            "motion_distance",
            "frame_gap_from_previous_ball",
        ]
        ordered = [column for column in preferred if column in df.columns]
        remaining = [column for column in df.columns if column not in ordered]
        output_df = df[ordered + remaining].copy()
        return output_df.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)

    def _records_by_frame(self, df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
        """시각화 속도를 위해 DataFrame row를 frame별 dict 리스트로 묶습니다."""

        if df.empty or "frame" not in df.columns:
            return {}
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in df.to_dict("records"):
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError):
                continue
            grouped.setdefault(frame, []).append(row)
        return grouped

    def _center_point(self, record: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        """row에서 중심 좌표를 OpenCV drawing용 int point로 변환합니다."""

        try:
            x = float(record["x_center"])
            y = float(record["y_center"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return int(round(x)), int(round(y))

    def _draw_status_box(self, frame: Any, frame_id: int, status: str) -> None:
        """검증 영상 좌상단에 현재 프레임과 공 상태를 표시합니다."""

        text = f"Frame: {frame_id} | Ball: {status}"
        cv2.rectangle(frame, (12, 12), (430, 58), (0, 0, 0), -1)
        cv2.putText(frame, text, (24, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    def _ball_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame에서 공 클래스 row만 추출합니다."""

        if df.empty or "class" not in df.columns:
            return df.iloc[0:0].copy()
        return df[df["class"].isin(BALL_CLASS_NAMES)].copy()

    def _ensure_numeric(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """좌표/프레임/confidence 컬럼을 안전하게 숫자로 변환합니다."""

        output_df = df.copy()
        for column in columns:
            if column in output_df.columns:
                output_df[column] = pd.to_numeric(output_df[column], errors="coerce")
        return output_df

    def _write_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """CSV 저장 공통 함수입니다."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, encoding="utf-8-sig")


def write_ball_quality_report(
    cleaned_df: pd.DataFrame,
    total_frames: Optional[int],
    output_path: str,
) -> pd.DataFrame:
    """정제된 공 좌표의 품질을 빠르게 검수할 수 있는 1행 리포트를 저장한다.

    이 리포트는 통계 계산 결과가 아니라 데이터 무결성 진단용이다.
    팀원은 coverage_pct, unreliable_ball_rows, 큰 점프 개수를 보고
    해당 영상이 슛/패스/리바운드 계산에 바로 쓸 수 있는지 먼저 판단할 수 있다.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ball_df = cleaned_df[cleaned_df.get("class", pd.Series(dtype=str)).eq("ball")].copy()
    frame_total = int(total_frames or 0)
    if ball_df.empty:
        report_df = pd.DataFrame(
            [
                {
                    "total_frames": frame_total,
                    "ball_rows": 0,
                    "ball_detected_frames": 0,
                    "ball_frame_coverage_pct": 0.0,
                    "reliable_ball_rows": 0,
                    "unreliable_ball_rows": 0,
                    "reliable_pct": 0.0,
                    "max_jump_distance": None,
                    "jump_gt_120_count": 0,
                    "jump_gt_180_count": 0,
                }
            ]
        )
        report_df.to_csv(output, index=False, encoding="utf-8-sig")
        return report_df

    frames = pd.to_numeric(ball_df.get("frame"), errors="coerce").dropna()
    jumps = pd.to_numeric(ball_df.get("jump_distance"), errors="coerce").dropna()
    reliable = ball_df.get("is_reliable", pd.Series([True] * len(ball_df), index=ball_df.index)).fillna(True).astype(bool)
    statuses = ball_df.get("ball_status", pd.Series(dtype=str)).fillna("Unknown").astype(str)

    # 프레임 커버리지는 "전체 영상 중 공 좌표가 존재하는 프레임 비율"이다.
    # 이 값이 낮으면 모델 탐지/보간이 부족하다는 뜻이고, 너무 높지만 unreliable이 많으면 false positive가 섞였을 가능성이 크다.
    coverage = (frames.nunique() / frame_total * 100.0) if frame_total > 0 else 0.0
    report: Dict[str, Any] = {
        "total_frames": frame_total,
        "ball_rows": int(len(ball_df)),
        "ball_detected_frames": int(frames.nunique()),
        "ball_frame_coverage_pct": round(coverage, 2),
        "reliable_ball_rows": int(reliable.sum()),
        "unreliable_ball_rows": int((~reliable).sum()),
        "reliable_pct": round(float(reliable.mean() * 100.0), 2) if len(reliable) > 0 else 0.0,
        "max_jump_distance": round(float(jumps.max()), 3) if not jumps.empty else None,
        "jump_gt_120_count": int(jumps.gt(120.0).sum()) if not jumps.empty else 0,
        "jump_gt_180_count": int(jumps.gt(180.0).sum()) if not jumps.empty else 0,
    }
    for status, count in statuses.value_counts().items():
        report[f"status_{status}"] = int(count)

    report_df = pd.DataFrame([report])
    report_df.to_csv(output, index=False, encoding="utf-8-sig")
    return report_df


def _safe_output_prefix(prefix: str) -> str:
    """여러 영상을 처리할 때 파일명이 서로 덮어쓰이지 않도록 prefix를 정리한다."""

    if not prefix:
        return ""
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(prefix))
    return safe if safe.endswith("_") else f"{safe}_"


def run_ball_coordinate_refinement(
    raw_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    video_path: str,
    output_dir: str,
    fps: float,
    frame_size: Tuple[int, int],
    total_frames: Optional[int],
    start_frame: int = 0,
    output_prefix: str = "",
    config: Optional[BallRefinerConfig] = None,
) -> Dict[str, str]:
    """파이프라인에서 한 번에 호출하기 위한 편의 함수입니다."""

    output = Path(output_dir)
    refiner = BallCoordinateRefiner(config or build_adaptive_refiner_config(frame_size, fps))
    prefix = _safe_output_prefix(output_prefix)
    raw_verify_path = str(output / f"{prefix}raw_detection_verify.csv")
    cleaned_path = str(output / f"{prefix}cleaned_tracking_results.csv")
    verification_video_path = str(output / f"{prefix}output_verification.mp4")
    quality_report_path = str(output / f"{prefix}ball_quality_report.csv")

    raw_verify_df = refiner.export_raw_detection_verify(raw_df, raw_verify_path)
    cleaned_df = refiner.create_cleaned_tracking_results(
        tracking_df=tracking_df,
        frame_size=frame_size,
        total_frames=total_frames,
        output_path=cleaned_path,
    )
    refiner.render_verification_video(
        video_path=video_path,
        raw_verify_df=raw_verify_df,
        cleaned_df=cleaned_df,
        output_path=verification_video_path,
        fps=fps,
        frame_size=frame_size,
        max_frames=total_frames,
        start_frame=start_frame,
    )
    write_ball_quality_report(
        cleaned_df=cleaned_df,
        total_frames=total_frames,
        output_path=quality_report_path,
    )

    return {
        "raw_detection_verify": raw_verify_path,
        "cleaned_tracking_results": cleaned_path,
        "verification_video": verification_video_path,
        "quality_report": quality_report_path,
    }


if __name__ == "__main__":
    # 단독 실행용 예시:
    # 이미 runs/detect/tracking_results.csv가 있을 때, 같은 파일을 raw/final 입력으로 사용해
    # 정제 CSV와 검증 영상을 다시 만들 수 있습니다.
    input_csv = "runs/detect/tracking_results.csv"
    input_video = "Video Project.mp4"
    df = pd.read_csv(input_csv)
    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps_value = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width_value = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_value = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    run_ball_coordinate_refinement(
        raw_df=df,
        tracking_df=df,
        video_path=input_video,
        output_dir="runs/detect",
        fps=fps_value,
        frame_size=(width_value, height_value),
        total_frames=frame_count_value,
    )
