"""농구공 검출 결과를 보정하는 후처리 모듈.

주요 기능:
- Kalman Filter로 공의 위치와 속도를 계속 추적한다.
- 공이 잠깐 사라진 구간은 선형/포물선 보간으로 메운다.
- 공 후보가 여러 개일 때 이전 궤적과 가장 자연스럽게 이어지는 후보를 고른다.
- 공이 선수와 겹쳐 안 보이는 프레임은 occlusion으로 표시하고 예측 좌표를 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Point = Dict[str, Any]


# 팀원용 역할 요약:
# 이 파일은 YOLO가 찾은 여러 공 후보 중 실제 공 궤적에 가장 가까운 좌표를 고르는 모듈입니다.
# 주요 단계는 1) 후보 선택, 2) 짧은 누락 구간 보간, 3) Kalman Filter smoothing,
# 4) 선수 가림(occlusion) 상황 표시입니다.
# video_yolo.py가 직접 이 파일을 호출하지 않고, tracking_pipeline.py -> basketball_postprocess.py를 통해 연결됩니다.


@dataclass
class BallTracker:
    """농구공 1개의 위치를 Kalman Filter로 추적한다.

    상태 벡터는 [x, y, vx, vy]이다.
    영상 좌표계에서는 y가 아래로 증가하므로, 양수 중력값은 공을 아래로 끌어당긴다.
    """

    # dt는 프레임 간 시간 간격이다. 일반 영상 후처리에서는 1프레임 단위로 둔다.
    dt: float = 1.0
    # process_noise가 클수록 급격한 방향 전환/속도 변화에 더 잘 따라간다.
    process_noise: float = 30.0
    # measurement_noise가 클수록 YOLO 측정값을 덜 믿고 예측값을 더 믿는다.
    measurement_noise: float = 12.0
    # 픽셀/frame^2 단위의 세로 가속도. 공의 포물선 예측에 사용한다.
    gravity_px_per_frame2: float = 0.0
    # 공이 너무 오래 사라지면 잘못된 예측이 누적되므로 추적을 리셋한다.
    max_missing: int = 30
    state: Optional[np.ndarray] = field(init=False, default=None)
    covariance: Optional[np.ndarray] = field(init=False, default=None)
    missing_frames: int = field(init=False, default=0)

    @property
    def initialized(self) -> bool:
        return self.state is not None and self.covariance is not None

    def reset(self) -> None:
        """현재 추적 상태를 모두 초기화한다."""
        self.state = None
        self.covariance = None
        self.missing_frames = 0

    def _transition(self, dt: Optional[float] = None) -> np.ndarray:
        """등속 운동 모델의 상태 전이 행렬을 만든다."""
        step = self.dt if dt is None else float(dt)
        return np.array(
            [
                [1.0, 0.0, step, 0.0],
                [0.0, 1.0, 0.0, step],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _gravity_control(self, dt: Optional[float] = None) -> np.ndarray:
        """포물선 운동을 반영하기 위한 중력 보정 벡터를 만든다."""
        step = self.dt if dt is None else float(dt)
        return np.array(
            [
                0.0,
                0.5 * self.gravity_px_per_frame2 * step * step,
                0.0,
                self.gravity_px_per_frame2 * step,
            ],
            dtype=float,
        )

    def _process_noise(self, dt: Optional[float] = None) -> np.ndarray:
        """모델이 설명하지 못하는 움직임 변화를 covariance에 반영한다."""
        step = self.dt if dt is None else float(dt)
        dt2 = step * step
        dt3 = dt2 * step
        dt4 = dt2 * dt2
        q = float(self.process_noise)
        return q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=float,
        )

    def _initialize(self, x: float, y: float) -> None:
        """첫 공 위치가 들어왔을 때 추적 상태를 만든다."""
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.covariance = np.diag([400.0, 400.0, 250.0, 250.0]).astype(float)
        self.missing_frames = 0

    def predict(self, dt: Optional[float] = None) -> Optional[Point]:
        """공이 검출되지 않은 프레임에서 다음 위치를 예측한다."""
        if not self.initialized:
            return None

        assert self.state is not None
        assert self.covariance is not None

        transition = self._transition(dt)
        # 이전 상태에 속도와 중력 효과를 더해서 현재 프레임 위치를 예측한다.
        self.state = transition @ self.state + self._gravity_control(dt)
        self.covariance = transition @ self.covariance @ transition.T + self._process_noise(dt)
        self.missing_frames += 1

        if self.missing_frames > self.max_missing:
            self.reset()
            return None

        return self.as_dict(source="kalman_prediction")

    def update(
        self,
        x: float,
        y: float,
        confidence: Optional[float] = None,
        measurement_noise: Optional[float] = None,
    ) -> Point:
        """YOLO/보간 측정값을 Kalman 상태에 반영한다."""
        if not self.initialized:
            self._initialize(float(x), float(y))
            return self.as_dict(source="kalman_update")

        assert self.state is not None
        assert self.covariance is not None

        measurement = np.array([float(x), float(y)], dtype=float)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

        conf = 1.0 if confidence is None else max(0.05, min(1.0, float(confidence)))
        noise = self.measurement_noise if measurement_noise is None else float(measurement_noise)
        # confidence가 낮은 탐지는 measurement variance를 키워서 덜 믿는다.
        measurement_variance = (noise**2) / conf
        measurement_covariance = np.eye(2, dtype=float) * measurement_variance

        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        # Kalman gain은 예측값과 측정값 중 어느 쪽을 더 믿을지 결정한다.
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        self.covariance = (identity - kalman_gain @ observation) @ self.covariance
        self.missing_frames = 0
        return self.as_dict(source="kalman_update")

    def step(
        self,
        measurement: Optional[Tuple[float, float]],
        confidence: Optional[float] = None,
        dt: Optional[float] = None,
        measurement_noise: Optional[float] = None,
    ) -> Optional[Point]:
        """한 프레임을 처리한다. 측정값이 있으면 update, 없으면 predict를 수행한다."""
        if measurement is None:
            return self.predict(dt)

        if self.initialized:
            prediction = self.predict(dt)
            if prediction is None:
                self._initialize(float(measurement[0]), float(measurement[1]))
                return self.as_dict(source="kalman_update")

        return self.update(
            float(measurement[0]),
            float(measurement[1]),
            confidence=confidence,
            measurement_noise=measurement_noise,
        )

    def as_dict(self, source: str) -> Point:
        if self.state is None:
            raise RuntimeError("BallTracker is not initialized.")
        return {
            "x_center": float(self.state[0]),
            "y_center": float(self.state[1]),
            "velocity_x": float(self.state[2]),
            "velocity_y": float(self.state[3]),
            "source": source,
            "missing_frames": self.missing_frames,
        }


def interpolate_ball_path(
    ball_points: Sequence[Mapping[str, Any]],
    max_gap: int = 18,
    method: str = "auto",
    gravity_px_per_frame2: Optional[float] = None,
    auto_gravity: bool = True,
    fallback_gravity_px_per_frame2: float = 0.8,
    min_gravity_px_per_frame2: float = 0.05,
    max_gravity_px_per_frame2: float = 3.0,
    support_window: int = 4,
    interpolated_confidence_floor: float = 0.75,
    total_frames: Optional[int] = None,
    frame_key: str = "frame",
    x_key: str = "x_center",
    y_key: str = "y_center",
    confidence_key: str = "confidence",
    rim_points_by_frame: Optional[Mapping[int, Sequence[Mapping[str, Any]]]] = None,
    rim_proximity_px: float = 180.0,
    rim_interpolation_extra_gap: int = 12,
) -> List[Point]:
    """짧게 끊긴 공 좌표 구간을 선형 또는 포물선 형태로 채운다."""

    # 짧게 사라진 공 좌표 구간만 보간합니다.
    # 긴 구간까지 억지로 이어 붙이면 슛/패스 분석에서 잘못된 공 위치가 생길 수 있습니다.
    if method not in {"auto", "linear", "quadratic"}:
        raise ValueError("method must be one of: auto, linear, quadratic")

    # 한 프레임에 공 후보가 여러 개 있으면 우선 confidence가 가장 높은 후보만 남긴다.
    best_by_frame = _select_best_point_by_frame(ball_points, frame_key, confidence_key)
    if not best_by_frame:
        return []

    # 전체 프레임 길이에 맞춰 detected/missing 상태를 가진 timeline을 만든다.
    max_seen_frame = max(best_by_frame)
    frame_count = max_seen_frame + 1 if total_frames is None else int(total_frames)
    timeline = [
        _make_timeline_row(frame, best_by_frame.get(frame), x_key, y_key, confidence_key)
        for frame in range(frame_count)
    ]

    valid_frames = [row["frame"] for row in timeline if _has_xy(row, x_key, y_key)]
    if len(valid_frames) < 2:
        return timeline

    # 전체 검출점에서 공 궤적에 사용할 중력값을 추정한다.
    global_gravity = _resolve_gravity(
        [timeline[frame] for frame in valid_frames],
        requested_gravity=gravity_px_per_frame2,
        auto_gravity=auto_gravity,
        fallback_gravity=fallback_gravity_px_per_frame2,
        min_gravity=min_gravity_px_per_frame2,
        max_gravity=max_gravity_px_per_frame2,
        frame_key=frame_key,
        y_key=y_key,
    )

    for left_frame, right_frame in zip(valid_frames, valid_frames[1:]):
        gap = right_frame - left_frame - 1
        if gap <= 0:
            continue

        # gap 주변 검출점만 따로 보고, 해당 구간에 더 어울리는 중력값을 다시 잡는다.
        left = timeline[left_frame]
        right = timeline[right_frame]
        # 공을 다시 잡은 지점이 이전 경로와 다른 segment이면, 두 점 사이를 억지로 잇지 않는다.
        # 이렇게 해야 엉뚱한 후보로 재시작했을 때 큰 직선/포물선 점프가 만들어지지 않는다.
        left_segment = left.get("ball_segment")
        right_segment = right.get("ball_segment")
        if left_segment is not None and right_segment is not None and left_segment != right_segment:
            continue
        near_rim_gap = _gap_near_rim(
            left,
            right,
            rim_points_by_frame,
            frame_key=frame_key,
            x_key=x_key,
            y_key=y_key,
            proximity_px=rim_proximity_px,
        )
        allowed_gap = (
            max_gap + max(0, int(rim_interpolation_extra_gap))
            if near_rim_gap
            else max_gap
        )
        if gap > allowed_gap:
            continue
        previous_point = _nearest_valid_before(timeline, left_frame, x_key, y_key)
        next_point = _nearest_valid_after(timeline, right_frame, x_key, y_key)
        support_points = _support_points_around_gap(
            timeline,
            left_frame,
            right_frame,
            support_window,
            x_key,
            y_key,
        )
        gap_gravity = _resolve_gravity(
            support_points,
            requested_gravity=gravity_px_per_frame2,
            auto_gravity=auto_gravity,
            fallback_gravity=global_gravity,
            min_gravity=min_gravity_px_per_frame2,
            max_gravity=max_gravity_px_per_frame2,
            frame_key=frame_key,
            y_key=y_key,
        )

        for frame in range(left_frame + 1, right_frame):
            # 누락 프레임의 x/y를 왼쪽 검출점과 오른쪽 검출점 사이에서 계산한다.
            interpolated_x, interpolated_y = _interpolate_between(
                frame,
                left,
                right,
                previous_point,
                next_point,
                method,
                gap_gravity,
                x_key,
                y_key,
            )
            # 보간점은 실제 탐지보다 약하지만, 궤적을 유지해야 하므로 최소 신뢰도를 준다.
            confidence = _interpolated_confidence(
                left,
                right,
                confidence_key,
                floor=interpolated_confidence_floor,
            )
            timeline[frame].update(
                {
                    x_key: interpolated_x,
                    y_key: interpolated_y,
                    confidence_key: confidence,
                    "detected": False,
                    "interpolated": True,
                    "measurement_source": "rim_interpolated" if near_rim_gap else "interpolated",
                    "interpolation_method": "linear" if method == "linear" else "quadratic",
                    "estimated_gravity": round(gap_gravity, 4),
                    "near_rim_gap": near_rim_gap,
                }
            )

    return timeline


def track_ball_detections(
    detections: Iterable[Mapping[str, Any]],
    total_frames: Optional[int] = None,
    max_gap: int = 18,
    interpolation: str = "auto",
    dt: float = 1.0,
    process_noise: float = 30.0,
    measurement_noise: float = 12.0,
    gravity_px_per_frame2: Optional[float] = None,
    auto_gravity: bool = True,
    fallback_gravity_px_per_frame2: float = 0.8,
    min_gravity_px_per_frame2: float = 0.05,
    max_gravity_px_per_frame2: float = 3.0,
    support_window: int = 4,
    trust_interpolated_arc: bool = True,
    interpolated_measurement_noise: float = 3.0,
    interpolated_confidence_floor: float = 0.75,
    max_prediction_frames: int = 30,
    enable_player_occlusion: bool = True,
    player_classes: Sequence[str] = ("person",),
    occlusion_margin_px: float = 35.0,
    player_center_radius_px: float = 90.0,
    enable_rim_proximity_boost: bool = True,
    rim_classes: Sequence[str] = ("hoop", "rim", "basketball hoop"),
    rim_proximity_px: float = 180.0,
    rim_interpolation_extra_gap: int = 12,
    use_motion_association: bool = True,
    association_max_distance_px: float = 220.0,
    association_gap_growth_px: float = 35.0,
    association_confidence_weight: float = 45.0,
    association_restart_after_frames: int = 18,
    association_restart_min_confidence: float = 0.03,
    ball_classes: Sequence[str] = ("sports ball", "frisbee", "basketball", "ball"),
    ball_label: str = "sports ball",
    stable_track_id: int = 0,
    class_key: str = "class",
    frame_key: str = "frame",
    x_key: str = "x_center",
    y_key: str = "y_center",
    confidence_key: str = "confidence",
) -> List[Point]:
    """YOLO 탐지 결과에서 보정된 농구공 좌표 시퀀스를 반환한다."""

    # 전체 객체 검출 row 중 공 후보만 따로 추려 단일 공 track으로 보정합니다.
    # 선수 row는 occlusion 판단에만 사용하고, 최종 선수 track 자체는 변경하지 않습니다.
    detection_list = list(detections)
    ball_class_set = set(ball_classes)
    # 공으로 볼 수 있는 클래스만 추려서 추적 대상으로 사용한다.
    ball_points = [
        item
        for item in detection_list
        if item.get(class_key) in ball_class_set
        and _as_float(item.get(x_key)) is not None
        and _as_float(item.get(y_key)) is not None
    ]
    # 선수 occlusion 판정을 빠르게 하기 위해 선수 탐지를 프레임별로 묶는다.
    players_by_frame = _group_detections_by_frame(
        detection_list,
        frame_key=frame_key,
        class_key=class_key,
        classes=player_classes,
    )
    rims_by_frame = _group_detections_by_frame(
        detection_list,
        frame_key=frame_key,
        class_key=class_key,
        classes=rim_classes,
    )

    if total_frames is None:
        frames = [
            _as_int(item.get(frame_key))
            for item in detection_list
            if _as_int(item.get(frame_key)) is not None
        ]
        total_frames = max(frames) + 1 if frames else None

    if use_motion_association:
        # 같은 프레임에 공 후보가 여러 개 있으면 이전 궤적과 가장 자연스럽게 이어지는 후보를 선택한다.
        ball_points = _select_ball_points_by_motion(
            ball_points,
            total_frames=total_frames,
            frame_key=frame_key,
            x_key=x_key,
            y_key=y_key,
            confidence_key=confidence_key,
            max_distance_px=association_max_distance_px,
            gap_growth_px=association_gap_growth_px,
            confidence_weight=association_confidence_weight,
            restart_after_frames=association_restart_after_frames,
            restart_min_confidence=association_restart_min_confidence,
        )

    # Kalman 예측과 포물선 보간에 사용할 중력값을 전체 공 후보에서 추정한다.
    tracker_gravity = _resolve_gravity(
        ball_points,
        requested_gravity=gravity_px_per_frame2,
        auto_gravity=auto_gravity,
        fallback_gravity=fallback_gravity_px_per_frame2,
        min_gravity=min_gravity_px_per_frame2,
        max_gravity=max_gravity_px_per_frame2,
        frame_key=frame_key,
        y_key=y_key,
    )

    # 짧은 미검출 구간을 먼저 보간해서 Kalman이 더 안정적인 입력을 받도록 한다.
    timeline = interpolate_ball_path(
        ball_points,
        max_gap=max_gap,
        method=interpolation,
        gravity_px_per_frame2=gravity_px_per_frame2,
        auto_gravity=auto_gravity,
        fallback_gravity_px_per_frame2=fallback_gravity_px_per_frame2,
        min_gravity_px_per_frame2=min_gravity_px_per_frame2,
        max_gravity_px_per_frame2=max_gravity_px_per_frame2,
        support_window=support_window,
        interpolated_confidence_floor=interpolated_confidence_floor,
        total_frames=total_frames,
        frame_key=frame_key,
        x_key=x_key,
        y_key=y_key,
        confidence_key=confidence_key,
        rim_points_by_frame=rims_by_frame if enable_rim_proximity_boost else None,
        rim_proximity_px=rim_proximity_px,
        rim_interpolation_extra_gap=rim_interpolation_extra_gap,
    )

    # 보간된 timeline을 다시 Kalman Filter로 통과시켜 좌표 흔들림을 줄인다.
    tracker = BallTracker(
        dt=dt,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        gravity_px_per_frame2=tracker_gravity,
        max_missing=max_prediction_frames,
    )

    tracked: List[Point] = []
    active_ball_segment: Optional[int] = None
    for row in timeline:
        # 측정값이 있으면 update, 없으면 predict만 수행한다.
        measurement = None
        if _has_xy(row, x_key, y_key):
            measurement = (float(row[x_key]), float(row[y_key]))

        row_segment = _as_int(row.get("ball_segment"))
        if measurement is not None and row_segment is not None:
            if active_ball_segment is not None and row_segment != active_ball_segment:
                # 다른 segment로 재획득된 공은 이전 Kalman 속도를 버리고 새 공 위치에서 바로 시작한다.
                # 그래야 이전 예측 위치에서 새 후보까지 느리게 끌려가거나 순간적으로 튀는 현상이 줄어든다.
                tracker = BallTracker(
                    dt=dt,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                    gravity_px_per_frame2=tracker_gravity,
                    max_missing=max_prediction_frames,
                )
            active_ball_segment = row_segment

        measurement_source = row.get("measurement_source", "missing")
        detected = bool(row.get("detected", False))
        interpolated = bool(row.get("interpolated", False))
        predicted = measurement is None
        effective_confidence = _as_float(row.get(confidence_key))
        measurement_noise_override = None
        if trust_interpolated_arc and interpolated:
            # 포물선 보간점은 궤적 형태를 유지해야 하므로 일반 탐지보다 더 강하게 반영한다.
            effective_confidence = max(effective_confidence or 0.0, interpolated_confidence_floor)
            measurement_noise_override = interpolated_measurement_noise

        filtered = tracker.step(
            measurement,
            confidence=effective_confidence,
            measurement_noise=measurement_noise_override,
        )
        if filtered is None:
            continue
        estimated_gravity = _as_float(row.get("estimated_gravity"))
        if estimated_gravity is None:
            estimated_gravity = tracker_gravity
        # 공이 미검출인데 예측 위치가 선수 근처라면 선수에 가려진 것으로 표시한다.
        occlusion = _find_player_occlusion(
            point=(filtered["x_center"], filtered["y_center"]),
            player_rows=players_by_frame.get(int(row["frame"]), []),
            enabled=enable_player_occlusion and predicted,
            margin_px=occlusion_margin_px,
            center_radius_px=player_center_radius_px,
            x_key=x_key,
            y_key=y_key,
        )
        occluded = occlusion is not None
        if occluded:
            measurement_source = "occluded_by_player"

        tracked.append(
            {
                frame_key: int(row["frame"]),
                "track_id": stable_track_id,
                "raw_track_id": row.get("raw_track_id", row.get("track_id", -1)),
                class_key: ball_label,
                confidence_key: round(float(row.get(confidence_key) or 0.0), 3),
                x_key: round(filtered["x_center"], 2),
                y_key: round(filtered["y_center"], 2),
                "measurement_x": _round_or_none(row.get(x_key)),
                "measurement_y": _round_or_none(row.get(y_key)),
                "velocity_x": round(filtered["velocity_x"], 2),
                "velocity_y": round(filtered["velocity_y"], 2),
                "dx": None,
                "dy": None,
                "detected": detected,
                "interpolated": interpolated,
                "predicted": predicted,
                "occluded": occluded,
                "occluded_by_track_id": occlusion.get("track_id") if occlusion else None,
                "occlusion_distance": occlusion.get("distance") if occlusion else None,
                "candidate_count": int(row.get("candidate_count", 0)),
                "association_distance": _round_or_none(row.get("association_distance")),
                "association_score": _round_or_none(row.get("association_score")),
                "measurement_source": measurement_source,
                "tracking_source": filtered["source"],
                "missing_frames": int(filtered["missing_frames"]),
                "estimated_gravity": round(estimated_gravity, 4),
            }
        )

    _append_motion_delta(tracked, x_key, y_key)
    return tracked


def correct_detections_with_ball_tracking(
    detections_df: Any,
    total_frames: Optional[int] = None,
    **tracking_kwargs: Any,
) -> Any:
    """DataFrame에서 공 row만 보정된 공 좌표로 교체한다."""

    import pandas as pd

    # 이 함수는 DataFrame 전체를 받아서 공 row만 교체하는 공개 API입니다.
    # basketball_postprocess.py에서 court filter 이후 호출됩니다.

    class_key = tracking_kwargs.get("class_key", "class")
    frame_key = tracking_kwargs.get("frame_key", "frame")
    ball_classes = tuple(
        tracking_kwargs.get(
            "ball_classes",
            ("sports ball", "frisbee", "basketball", "ball"),
        )
    )

    # 전체 탐지 결과를 넘겨야 공뿐 아니라 선수 occlusion 정보도 함께 사용할 수 있다.
    tracked_ball = track_ball_detections(
        detections_df.to_dict("records"),
        total_frames=total_frames,
        **tracking_kwargs,
    )

    if not tracked_ball:
        return detections_df.copy()

    # 기존 공 후보 row는 제거하고, 안정화된 단일 공 궤적 row로 대체한다.
    non_ball_df = detections_df[~detections_df[class_key].isin(ball_classes)].copy()
    ball_df = pd.DataFrame(tracked_ball)
    combined = pd.concat([non_ball_df, ball_df], ignore_index=True, sort=False)

    sort_columns = [
        column
        for column in [frame_key, class_key, "track_id"]
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)

    return combined


def correct_detection_csv(
    input_csv: str = "video_detection.csv",
    output_csv: str = "video_detection_ball_tracked.csv",
    total_frames: Optional[int] = None,
    **tracking_kwargs: Any,
) -> Any:
    """video_yolo.py를 수정하지 않고 CSV만 읽어 공 좌표를 보정하는 wrapper."""

    import pandas as pd

    detections_df = pd.read_csv(input_csv)
    corrected_df = correct_detections_with_ball_tracking(
        detections_df,
        total_frames=total_frames,
        **tracking_kwargs,
    )
    corrected_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return corrected_df


def _predict_ball_candidate_position(
    previous: Optional[Mapping[str, Any]],
    latest: Mapping[str, Any],
    target_frame: int,
    frame_key: str,
    x_key: str,
    y_key: str,
) -> Tuple[float, float]:
    """직전 두 공 위치로 target_frame의 예상 위치를 계산한다."""
    latest_x = _as_float(latest.get(x_key)) or 0.0
    latest_y = _as_float(latest.get(y_key)) or 0.0
    latest_frame = _as_int(latest.get(frame_key)) or target_frame
    if previous is None:
        return latest_x, latest_y

    previous_x = _as_float(previous.get(x_key))
    previous_y = _as_float(previous.get(y_key))
    previous_frame = _as_int(previous.get(frame_key))
    if previous_x is None or previous_y is None or previous_frame is None:
        return latest_x, latest_y

    frame_delta = float(latest_frame - previous_frame)
    if frame_delta <= 0.0:
        return latest_x, latest_y

    step = float(target_frame - latest_frame)
    velocity_x = (latest_x - previous_x) / frame_delta
    velocity_y = (latest_y - previous_y) / frame_delta
    return latest_x + velocity_x * step, latest_y + velocity_y * step


def _estimate_candidate_speed(
    previous: Optional[Mapping[str, Any]],
    latest: Mapping[str, Any],
    frame_key: str,
    x_key: str,
    y_key: str,
) -> Optional[float]:
    """직전 두 공 위치 사이의 픽셀/frame 속도를 계산한다."""
    if previous is None:
        return None

    previous_x = _as_float(previous.get(x_key))
    previous_y = _as_float(previous.get(y_key))
    previous_frame = _as_int(previous.get(frame_key))
    latest_x = _as_float(latest.get(x_key))
    latest_y = _as_float(latest.get(y_key))
    latest_frame = _as_int(latest.get(frame_key))
    if None in (previous_x, previous_y, previous_frame, latest_x, latest_y, latest_frame):
        return None

    frame_delta = float(int(latest_frame) - int(previous_frame))
    if frame_delta <= 0.0:
        return None
    return (
        math.hypot(
            float(latest_x) - float(previous_x),
            float(latest_y) - float(previous_y),
        )
        / frame_delta
    )


def _select_best_point_by_frame(
    points: Sequence[Mapping[str, Any]],
    frame_key: str,
    confidence_key: str,
) -> Dict[int, Point]:
    """프레임별로 confidence가 가장 높은 공 후보를 선택한다."""
    best_by_frame: Dict[int, Point] = {}
    for point in points:
        frame = _as_int(point.get(frame_key))
        if frame is None:
            continue

        confidence = _as_float(point.get(confidence_key)) or 0.0
        current = best_by_frame.get(frame)
        current_confidence = _as_float(current.get(confidence_key)) if current is not None else None
        if current is None or confidence >= (current_confidence or 0.0):
            best_by_frame[frame] = dict(point)

    return best_by_frame


def _make_timeline_row(
    frame: int,
    point: Optional[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    confidence_key: str,
) -> Point:
    """한 프레임의 공 정보를 timeline row 형태로 만든다."""
    if point is None:
        return {
            "frame": frame,
            x_key: None,
            y_key: None,
            confidence_key: 0.0,
            "detected": False,
            "interpolated": False,
            "measurement_source": "missing",
        }

    row = dict(point)
    row["frame"] = frame
    row[x_key] = _as_float(row.get(x_key))
    row[y_key] = _as_float(row.get(y_key))
    row["raw_track_id"] = row.get("track_id", -1)
    row["detected"] = _has_xy(row, x_key, y_key)
    row["interpolated"] = False
    row["measurement_source"] = "detected" if row["detected"] else "missing"
    return row


def _interpolate_between(
    frame: int,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    previous_point: Optional[Mapping[str, Any]],
    next_point: Optional[Mapping[str, Any]],
    method: str,
    gravity_px_per_frame2: Optional[float],
    x_key: str,
    y_key: str,
) -> Tuple[float, float]:
    """두 검출점 사이의 누락 프레임 좌표를 선형 또는 포물선으로 계산한다."""
    left_frame = int(left["frame"])
    right_frame = int(right["frame"])
    total_time = float(right_frame - left_frame)
    elapsed = float(frame - left_frame)
    ratio = elapsed / total_time

    x0 = float(left[x_key])
    y0 = float(left[y_key])
    x1 = float(right[x_key])
    y1 = float(right[y_key])

    x = x0 + (x1 - x0) * ratio
    if method == "linear":
        # 단순 이동으로 충분한 짧은 구간은 선형 보간을 사용할 수 있다.
        y = y0 + (y1 - y0) * ratio
        return float(x), float(y)

    left_slope = _slope(previous_point, left, y_key)
    right_slope = _slope(right, next_point, y_key)
    y = _parabolic_y(
        y0,
        y1,
        total_time,
        elapsed,
        left_slope=left_slope,
        right_slope=right_slope,
        gravity_px_per_frame2=gravity_px_per_frame2,
    )
    return float(x), float(y)


def _parabolic_y(
    y0: float,
    y1: float,
    total_time: float,
    elapsed: float,
    left_slope: Optional[float],
    right_slope: Optional[float],
    gravity_px_per_frame2: Optional[float],
) -> float:
    """y축 위치를 포물선 공식으로 계산한다."""
    if total_time <= 0:
        return y0

    gravity = 0.0 if gravity_px_per_frame2 is None else float(gravity_px_per_frame2)
    if abs(gravity) > 1e-9:
        # 중력값이 있으면 y = y0 + v0*t + 0.5*g*t^2 공식을 사용한다.
        initial_velocity = (y1 - y0 - 0.5 * gravity * total_time * total_time) / total_time
        return y0 + initial_velocity * elapsed + 0.5 * gravity * elapsed * elapsed

    if left_slope is not None:
        acceleration = 2.0 * (y1 - y0 - left_slope * total_time) / (total_time * total_time)
        return y0 + left_slope * elapsed + 0.5 * acceleration * elapsed * elapsed

    if right_slope is not None:
        acceleration = 2.0 * (y0 - y1 + right_slope * total_time) / (total_time * total_time)
        initial_velocity = right_slope - acceleration * total_time
        return y0 + initial_velocity * elapsed + 0.5 * acceleration * elapsed * elapsed

    ratio = elapsed / total_time
    return y0 + (y1 - y0) * ratio


def _nearest_valid_before(
    timeline: Sequence[Mapping[str, Any]],
    start_frame: int,
    x_key: str,
    y_key: str,
) -> Optional[Mapping[str, Any]]:
    """현재 gap 왼쪽에서 가장 가까운 유효 공 좌표를 찾는다."""
    for index in range(start_frame - 1, -1, -1):
        if _has_xy(timeline[index], x_key, y_key):
            return timeline[index]
    return None


def _nearest_valid_after(
    timeline: Sequence[Mapping[str, Any]],
    start_frame: int,
    x_key: str,
    y_key: str,
) -> Optional[Mapping[str, Any]]:
    """현재 gap 오른쪽에서 가장 가까운 유효 공 좌표를 찾는다."""
    for index in range(start_frame + 1, len(timeline)):
        if _has_xy(timeline[index], x_key, y_key):
            return timeline[index]
    return None


def _support_points_around_gap(
    timeline: Sequence[Mapping[str, Any]],
    left_frame: int,
    right_frame: int,
    support_window: int,
    x_key: str,
    y_key: str,
) -> List[Mapping[str, Any]]:
    """gap 주변의 검출점을 모아 해당 구간의 포물선 추정에 사용한다."""
    start = max(0, left_frame - max(0, support_window))
    end = min(len(timeline), right_frame + max(0, support_window) + 1)
    return [row for row in timeline[start:end] if _has_xy(row, x_key, y_key)]


def _gap_near_rim(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    rim_points_by_frame: Optional[Mapping[int, Sequence[Mapping[str, Any]]]],
    frame_key: str,
    x_key: str,
    y_key: str,
    proximity_px: float,
) -> bool:
    """보간하려는 gap이 림 주변인지 판단한다.

    슛 궤적은 림/그물에 가까워질 때 공이 가려져 검출 confidence가 급락한다. 이때만
    gap 허용치를 늘리면 일반 패스 구간까지 과하게 이어 붙이지 않으면서 림 근처
    ID Lost를 줄일 수 있다.
    """
    if not rim_points_by_frame:
        return False

    left_frame = _as_int(left.get(frame_key))
    right_frame = _as_int(right.get(frame_key))
    left_x = _as_float(left.get(x_key))
    left_y = _as_float(left.get(y_key))
    right_x = _as_float(right.get(x_key))
    right_y = _as_float(right.get(y_key))
    if None in (left_frame, right_frame, left_x, left_y, right_x, right_y):
        return False

    middle_frame = int(round((int(left_frame) + int(right_frame)) / 2.0))
    middle_point = (
        (float(left_x) + float(right_x)) / 2.0,
        (float(left_y) + float(right_y)) / 2.0,
    )
    checks = [
        (int(left_frame), (float(left_x), float(left_y))),
        (middle_frame, middle_point),
        (int(right_frame), (float(right_x), float(right_y))),
    ]

    for frame, point in checks:
        rims = _rim_points_around_frame(rim_points_by_frame, frame, window=8)
        if _point_near_rims(point, rims, x_key, y_key, proximity_px):
            return True
    return False


def _rim_points_around_frame(
    rim_points_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    frame: int,
    window: int,
) -> List[Mapping[str, Any]]:
    """현재 프레임 주변의 림 탐지를 모은다."""
    rims: List[Mapping[str, Any]] = []
    for current_frame in range(frame - window, frame + window + 1):
        rims.extend(rim_points_by_frame.get(current_frame, []))
    return rims


def _point_near_rims(
    point: Tuple[float, float],
    rims: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    proximity_px: float,
) -> bool:
    """점이 림 bbox 또는 림 중심 근처에 있는지 확인한다."""
    if not rims:
        return False

    for rim in rims:
        bbox = _bbox_from_detection(rim, x_key=x_key, y_key=y_key)
        if bbox is not None and _distance_to_bbox(point, bbox) <= proximity_px:
            return True
        center = _center_from_detection(rim, x_key=x_key, y_key=y_key)
        if center is not None and math.hypot(
            point[0] - center[0],
            point[1] - center[1],
        ) <= proximity_px:
            return True
    return False


def _slope(
    left: Optional[Mapping[str, Any]],
    right: Optional[Mapping[str, Any]],
    y_key: str,
) -> Optional[float]:
    """두 점 사이의 y축 기울기를 계산한다."""
    if left is None or right is None:
        return None
    left_y = _as_float(left.get(y_key))
    right_y = _as_float(right.get(y_key))
    if left_y is None or right_y is None:
        return None

    frame_delta = float(int(right["frame"]) - int(left["frame"]))
    if frame_delta == 0.0:
        return None
    return (right_y - left_y) / frame_delta


def _resolve_gravity(
    points: Sequence[Mapping[str, Any]],
    requested_gravity: Optional[float],
    auto_gravity: bool,
    fallback_gravity: float,
    min_gravity: float,
    max_gravity: float,
    frame_key: str,
    y_key: str,
) -> float:
    """입력/자동추정/fallback 중 사용할 중력값을 결정한다."""
    requested = _as_float(requested_gravity)
    if requested is not None and abs(requested) > 1e-9:
        return _clamp_acceleration(requested, min_gravity, max_gravity)

    if auto_gravity:
        estimated = _estimate_gravity_from_points(points, frame_key, y_key)
        if estimated is not None and abs(estimated) > 1e-9:
            return _clamp_acceleration(estimated, min_gravity, max_gravity)

    fallback = _as_float(fallback_gravity)
    if fallback is None:
        return 0.0
    return _clamp_acceleration(fallback, min_gravity, max_gravity)


def _estimate_gravity_from_points(
    points: Sequence[Mapping[str, Any]],
    frame_key: str,
    y_key: str,
) -> Optional[float]:
    """검출점들의 y축 변화량으로 공의 세로 가속도를 추정한다."""
    valid_points: List[Tuple[int, float]] = []
    for point in points:
        frame = _as_int(point.get(frame_key))
        if frame is None:
            frame = _as_int(point.get("frame"))
        y = _as_float(point.get(y_key))
        if frame is not None and y is not None:
            valid_points.append((frame, y))

    unique_by_frame = {frame: y for frame, y in valid_points}
    ordered = sorted(unique_by_frame.items())
    if len(ordered) < 3:
        return None

    accelerations: List[float] = []
    for (frame0, y0), (frame1, y1), (frame2, y2) in zip(ordered, ordered[1:], ordered[2:]):
        dt0 = float(frame1 - frame0)
        dt1 = float(frame2 - frame1)
        if dt0 <= 0.0 or dt1 <= 0.0:
            continue

        velocity0 = (y1 - y0) / dt0
        velocity1 = (y2 - y1) / dt1
        acceleration = 2.0 * (velocity1 - velocity0) / (dt0 + dt1)
        if math.isfinite(acceleration):
            accelerations.append(acceleration)

    if not accelerations:
        return None

    # 영상 좌표에서는 아래 방향 가속도가 양수이므로, 양수 가속도를 우선 사용한다.
    positive_accelerations = [value for value in accelerations if value > 0.0]
    candidates = positive_accelerations if positive_accelerations else accelerations
    return float(np.median(candidates))


def _clamp_acceleration(value: float, min_abs: float, max_abs: float) -> float:
    """추정된 가속도가 너무 작거나 큰 값으로 튀지 않도록 제한한다."""
    if abs(value) <= 1e-9:
        return 0.0

    lower = max(0.0, float(min_abs))
    upper = max(lower, float(max_abs))
    sign = -1.0 if value < 0.0 else 1.0
    return sign * min(max(abs(float(value)), lower), upper)


def _group_detections_by_frame(
    detections: Sequence[Mapping[str, Any]],
    frame_key: str,
    class_key: str,
    classes: Sequence[str],
) -> Dict[int, List[Mapping[str, Any]]]:
    """특정 클래스 탐지를 프레임 번호별로 묶는다."""
    class_set = set(classes)
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for detection in detections:
        if detection.get(class_key) not in class_set:
            continue
        frame = _as_int(detection.get(frame_key))
        if frame is None:
            continue
        grouped.setdefault(frame, []).append(detection)
    return grouped


def _find_player_occlusion(
    point: Tuple[float, float],
    player_rows: Sequence[Mapping[str, Any]],
    enabled: bool,
    margin_px: float,
    center_radius_px: float,
    x_key: str,
    y_key: str,
) -> Optional[Point]:
    """예측된 공 위치가 선수 박스/중심 근처에 있는지 검사한다."""
    if not enabled or not player_rows:
        return None

    best: Optional[Point] = None
    best_distance: Optional[float] = None
    for player in player_rows:
        bbox = _bbox_from_detection(player, x_key=x_key, y_key=y_key)
        if bbox is not None:
            # bbox가 있으면 선수 사각형 내부 또는 margin 근처를 occlusion으로 본다.
            distance = _distance_to_bbox(point, bbox)
            if distance <= margin_px:
                best, best_distance = _better_occlusion_candidate(
                    player,
                    distance,
                    best,
                    best_distance,
                )
            continue

        center = _center_from_detection(player, x_key=x_key, y_key=y_key)
        if center is None:
            continue
        distance = math.hypot(point[0] - center[0], point[1] - center[1])
        if distance <= center_radius_px:
            # bbox가 없는 CSV에서는 선수 중심 반경으로 가림 여부를 추정한다.
            best, best_distance = _better_occlusion_candidate(
                player,
                distance,
                best,
                best_distance,
            )

    return best


def _better_occlusion_candidate(
    player: Mapping[str, Any],
    distance: float,
    best: Optional[Point],
    best_distance: Optional[float],
) -> Tuple[Optional[Point], Optional[float]]:
    """여러 선수 후보 중 공과 가장 가까운 선수를 선택한다."""
    if best_distance is not None and distance >= best_distance:
        return best, best_distance

    track_id = _as_int(player.get("track_id"))
    return {
        "track_id": track_id if track_id is not None else -1,
        "distance": round(float(distance), 2),
    }, distance


def _bbox_from_detection(
    detection: Mapping[str, Any],
    x_key: str,
    y_key: str,
) -> Optional[Tuple[float, float, float, float]]:
    """탐지 row에서 bbox 정보를 가능한 여러 컬럼명으로 읽어온다."""
    if isinstance(detection.get("bbox"), Sequence) and len(detection["bbox"]) >= 4:
        x1, y1, x2, y2 = detection["bbox"][:4]
        return float(x1), float(y1), float(x2), float(y2)

    keys = [
        ("x1", "y1", "x2", "y2"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    ]
    for x1_key, y1_key, x2_key, y2_key in keys:
        values = [
            _as_float(detection.get(x1_key)),
            _as_float(detection.get(y1_key)),
            _as_float(detection.get(x2_key)),
            _as_float(detection.get(y2_key)),
        ]
        if all(value is not None for value in values):
            x1, y1, x2, y2 = [float(value) for value in values]
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    width = _as_float(detection.get("width"))
    height = _as_float(detection.get("height"))
    center = _center_from_detection(detection, x_key=x_key, y_key=y_key)
    if width is not None and height is not None and center is not None:
        half_width = width / 2.0
        half_height = height / 2.0
        return (
            center[0] - half_width,
            center[1] - half_height,
            center[0] + half_width,
            center[1] + half_height,
        )

    return None


def _center_from_detection(
    detection: Mapping[str, Any],
    x_key: str,
    y_key: str,
) -> Optional[Tuple[float, float]]:
    """탐지 row의 중심 좌표를 읽는다."""
    x = _as_float(detection.get(x_key))
    y = _as_float(detection.get(y_key))
    if x is None or y is None:
        return None
    return x, y


def _distance_to_bbox(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> float:
    """점이 bbox에서 얼마나 떨어져 있는지 계산한다. 내부면 0이다."""
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _interpolated_confidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    confidence_key: str,
    floor: float = 0.75,
) -> float:
    """보간점에 줄 confidence를 계산한다."""
    left_confidence = _as_float(left.get(confidence_key))
    right_confidence = _as_float(right.get(confidence_key))
    confidences = [value for value in [left_confidence, right_confidence] if value is not None]
    if not confidences:
        return floor
    return max(0.05, min(1.0, max(floor, min(confidences) * 0.8)))


def _append_motion_delta(records: List[Point], x_key: str, y_key: str) -> None:
    """연속 프레임 간 dx/dy 이동량을 결과 row에 추가한다."""
    previous_x: Optional[float] = None
    previous_y: Optional[float] = None
    for record in records:
        current_x = _as_float(record.get(x_key))
        current_y = _as_float(record.get(y_key))
        if previous_x is None or previous_y is None or current_x is None or current_y is None:
            record["dx"] = None
            record["dy"] = None
        else:
            record["dx"] = round(current_x - previous_x, 2)
            record["dy"] = round(current_y - previous_y, 2)
        previous_x = current_x
        previous_y = current_y


def _has_xy(row: Mapping[str, Any], x_key: str, y_key: str) -> bool:
    """row에 유효한 x/y 좌표가 있는지 확인한다."""
    return _as_float(row.get(x_key)) is not None and _as_float(row.get(y_key)) is not None


def _as_float(value: Any) -> Optional[float]:
    """값을 float로 안전하게 변환한다. NaN/inf는 None으로 처리한다."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _as_int(value: Any) -> Optional[int]:
    """값을 int로 안전하게 변환한다."""
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _round_or_none(value: Any, ndigits: int = 2) -> Optional[float]:
    """숫자는 반올림하고, 유효하지 않은 값은 None으로 둔다."""
    number = _as_float(value)
    if number is None:
        return None
    return round(number, ndigits)


def _select_ball_points_by_motion(
    points: Sequence[Mapping[str, Any]],
    total_frames: Optional[int],
    frame_key: str,
    x_key: str,
    y_key: str,
    confidence_key: str,
    max_distance_px: float,
    gap_growth_px: float,
    confidence_weight: float,
    restart_after_frames: int,
    restart_min_confidence: float,
) -> List[Point]:
    """긴 후보 공백을 나눠 각 구간에서 그럴듯한 공 경로를 하나씩 남긴다.

    긴 영상에서는 초반, 중반, 후반의 조명/카메라 구도가 달라질 수 있다. 전체를
    하나의 경로로만 고르면 후반부의 강한 후보가 초반 공을 밀어내는 문제가 생긴다.
    그래서 공 후보가 오래 끊긴 지점에서 chunk를 나누고, 각 chunk를 독립적으로 고른다.
    """

    # 긴 영상 전체를 한 번에 최적화하지 않고 chunk 단위로 나누어 처리합니다.
    # 각 chunk 안에서는 beam search로 여러 후보 경로를 비교하고 가장 자연스러운 공 경로를 선택합니다.
    if not points:
        return []

    # restart_after_frames를 그대로 chunk 기준으로 사용한다. 별도 탐색을 추가하지
    # 않아 후처리 비용을 예측 가능하게 유지하면서 긴 영상의 구간별 공 탐지를 살린다.
    split_gap = max(1, restart_after_frames)
    chunks = _split_motion_points_by_gap(points, frame_key=frame_key, max_gap=split_gap)
    selected: List[Point] = []
    segment_offset = 0

    for chunk in chunks:
        chunk_rows = _select_ball_points_by_motion_chunk(
            chunk,
            total_frames=total_frames,
            frame_key=frame_key,
            x_key=x_key,
            y_key=y_key,
            confidence_key=confidence_key,
            max_distance_px=max_distance_px,
            gap_growth_px=gap_growth_px,
            confidence_weight=confidence_weight,
            restart_after_frames=restart_after_frames,
            restart_min_confidence=restart_min_confidence,
        )
        if not chunk_rows:
            continue

        max_segment = 0
        for row in chunk_rows:
            output_row = dict(row)
            segment_id = _as_int(output_row.get("ball_segment"))
            if segment_id is None:
                segment_id = 0
            output_row["ball_segment"] = int(segment_id + segment_offset)
            max_segment = max(max_segment, int(segment_id))
            selected.append(output_row)
        segment_offset += max_segment + 1

    return sorted(
        selected,
        key=lambda row: (
            _as_int(row.get(frame_key)) or 0,
            _as_float(row.get(confidence_key)) or 0.0,
        ),
    )


def _select_ball_points_by_motion_chunk(
    points: Sequence[Mapping[str, Any]],
    total_frames: Optional[int],
    frame_key: str,
    x_key: str,
    y_key: str,
    confidence_key: str,
    max_distance_px: float,
    gap_growth_px: float,
    confidence_weight: float,
    restart_after_frames: int,
    restart_min_confidence: float,
) -> List[Point]:
    """여러 공 후보 중 전체 구간에서 가장 자연스러운 공 경로를 고른다.

    기존 greedy 방식은 한 프레임에서 한 번 잘못 고르면 그 위치를 기준으로 다음 프레임을
    예측하기 때문에 공이 튀거나 정상 공으로 늦게 돌아오는 문제가 있었다. 이 함수는 여러
    후보 경로를 동시에 유지하는 beam search로 짧은 미래 구간까지 함께 보고 선택한다.
    """
    if not points:
        return []

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for point in points:
        frame = _as_int(point.get(frame_key))
        if frame is None:
            continue
        grouped.setdefault(frame, []).append(point)

    if not grouped:
        return []

    frame_candidate_counts = {frame: len(candidates) for frame, candidates in grouped.items()}
    grouped = {
        frame: _top_motion_candidates(candidates, confidence_key, x_key, y_key, limit=12)
        for frame, candidates in grouped.items()
    }
    frame_order = sorted(grouped)

    # beam_width가 클수록 후보를 더 오래 비교하지만 느려진다. 18개 정도면 농구공 후보가
    # 산발적으로 여러 개 나오는 상황에서도 속도와 안정성의 균형이 좋다.
    beam_width = 18
    max_link_gap = max(1, restart_after_frames)
    max_keep_gap = max(max_link_gap, 90)
    restart_gap = max(3, min(restart_after_frames, 8))
    restart_confidence_floor = max(restart_min_confidence, 0.08)
    beam: List[Dict[str, Any]] = []

    for frame in frame_order:
        candidates = grouped.get(frame, [])
        if not candidates:
            continue

        active_beam = _active_motion_hypotheses(
            beam,
            frame=frame,
            frame_key=frame_key,
            max_link_gap=max_keep_gap,
        )
        expanded: List[Dict[str, Any]] = []
        candidate_count = frame_candidate_counts.get(frame, len(candidates))

        for candidate in candidates:
            confidence = _as_float(candidate.get(confidence_key)) or 0.0
            if confidence < restart_min_confidence:
                continue

            # 어느 프레임에서든 새 경로를 시작할 수 있게 둔다. 그래야 초반에 공을 놓쳤거나
            # 이전 경로가 틀어졌을 때 정상 후보로 빠르게 갈아탈 수 있다.
            start_row = _make_motion_candidate_row(
                candidate,
                candidate_count=candidate_count,
                segment_id=0,
                association_distance=0.0,
                association_score=0.0,
                association_mode="start",
            )
            expanded.append(
                {
                    "score": (
                        _motion_candidate_quality(candidate, confidence_key, x_key, y_key)
                        + 0.8
                    ),
                    "hits": 1,
                    "restarts": 0,
                    "segment_id": 0,
                    "previous": None,
                    "latest": start_row,
                    "path": [start_row],
                }
            )

            for hypothesis in active_beam:
                latest = hypothesis["latest"]
                previous = hypothesis.get("previous")
                transition = _score_motion_transition(
                    previous=previous,
                    latest=latest,
                    candidate=candidate,
                    target_frame=frame,
                    frame_key=frame_key,
                    x_key=x_key,
                    y_key=y_key,
                    confidence_key=confidence_key,
                    max_distance_px=max_distance_px,
                    gap_growth_px=gap_growth_px,
                    confidence_weight=confidence_weight,
                )

                if transition is not None:
                    row = _make_motion_candidate_row(
                        candidate,
                        candidate_count=candidate_count,
                        segment_id=int(hypothesis.get("segment_id", 0)),
                        association_distance=transition["distance"],
                        association_score=transition["score"],
                        association_mode="linked",
                    )
                    expanded.append(
                        {
                            "score": float(hypothesis["score"]) + transition["score"],
                            "hits": int(hypothesis["hits"]) + 1,
                            "restarts": int(hypothesis.get("restarts", 0)),
                            "segment_id": int(hypothesis.get("segment_id", 0)),
                            "previous": latest,
                            "latest": row,
                            "path": hypothesis["path"] + [row],
                        }
                    )
                    continue

                latest_frame = _as_int(latest.get(frame_key))
                frame_gap = frame - latest_frame if latest_frame is not None else 0
                if frame_gap < restart_gap or confidence < restart_confidence_floor:
                    continue

                # 일정 프레임 이상 공을 놓친 뒤에는 이전 속도 예측을 버리고 재시작한다.
                # segment_id를 바꿔 보간 단계에서 이전 위치와 억지로 이어지지 않게 한다.
                next_segment_id = int(hypothesis.get("segment_id", 0)) + 1
                restart_row = _make_motion_candidate_row(
                    candidate,
                    candidate_count=candidate_count,
                    segment_id=next_segment_id,
                    association_distance=0.0,
                    association_score=0.0,
                    association_mode="restart",
                )
                restart_score = (
                    _motion_candidate_quality(candidate, confidence_key, x_key, y_key)
                    + 0.45
                    - 2.0
                    - min(2.0, max(0, frame_gap - restart_gap) * 0.12)
                )
                expanded.append(
                    {
                        "score": float(hypothesis["score"]) + restart_score,
                        "hits": int(hypothesis["hits"]) + 1,
                        "restarts": int(hypothesis.get("restarts", 0)) + 1,
                        "segment_id": next_segment_id,
                        "previous": None,
                        "latest": restart_row,
                        "path": hypothesis["path"] + [restart_row],
                    }
                )

        if expanded:
            beam = sorted(
                expanded + active_beam,
                key=_rank_motion_hypothesis,
                reverse=True,
            )[:beam_width]
        else:
            beam = active_beam[:beam_width]

    if not beam:
        return []

    best = max(beam, key=_rank_motion_hypothesis)
    return [dict(row) for row in best["path"]]


def _split_motion_points_by_gap(
    points: Sequence[Mapping[str, Any]],
    frame_key: str,
    max_gap: int,
) -> List[List[Mapping[str, Any]]]:
    """공 후보가 오래 비는 지점에서 후보 목록을 여러 구간으로 나눈다."""

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for point in points:
        frame = _as_int(point.get(frame_key))
        if frame is None:
            continue
        grouped.setdefault(frame, []).append(point)

    chunks: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    previous_frame: Optional[int] = None
    for frame in sorted(grouped):
        if previous_frame is not None and frame - previous_frame > max_gap:
            if current:
                chunks.append(current)
            current = []
        current.extend(grouped[frame])
        previous_frame = frame

    if current:
        chunks.append(current)
    return chunks


def _top_motion_candidates(
    candidates: Sequence[Mapping[str, Any]],
    confidence_key: str,
    x_key: str,
    y_key: str,
    limit: int,
) -> List[Mapping[str, Any]]:
    """한 프레임에 후보가 너무 많을 때 공답지 않은 후보를 먼저 줄인다."""
    valid_candidates = [
        candidate
        for candidate in candidates
        if _as_float(candidate.get(x_key)) is not None
        and _as_float(candidate.get(y_key)) is not None
    ]
    return sorted(
        valid_candidates,
        key=lambda item: _motion_candidate_quality(item, confidence_key, x_key, y_key),
        reverse=True,
    )[:limit]


def _motion_candidate_quality(
    candidate: Mapping[str, Any],
    confidence_key: str,
    x_key: str,
    y_key: str,
) -> float:
    """confidence와 bbox 모양을 함께 보고 공 후보의 기본 품질 점수를 계산한다."""
    confidence = _as_float(candidate.get(confidence_key)) or 0.0
    quality = confidence * 4.0

    bbox = _bbox_from_detection(candidate, x_key=x_key, y_key=y_key)
    if bbox is None:
        return quality - 0.25

    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return quality - 0.75

    area = width * height
    aspect = max(width / height, height / width)

    # 공 bbox는 대체로 작고 둥글다. 손/신발/글자 조각처럼 길쭉하거나 큰 후보는 감점한다.
    quality -= max(0.0, aspect - 1.35) * 0.65
    quality -= max(0.0, area - 2800.0) / 2200.0
    quality -= max(0.0, 18.0 - area) / 60.0
    return quality


def _active_motion_hypotheses(
    beam: Sequence[Mapping[str, Any]],
    frame: int,
    frame_key: str,
    max_link_gap: int,
) -> List[Dict[str, Any]]:
    """현재 프레임까지 이어질 가능성이 남아 있는 후보 경로만 유지한다."""
    active: List[Dict[str, Any]] = []
    for hypothesis in beam:
        latest = hypothesis.get("latest")
        if not isinstance(latest, Mapping):
            continue
        latest_frame = _as_int(latest.get(frame_key))
        if latest_frame is None:
            continue
        frame_gap = frame - latest_frame
        if frame_gap <= 0 or frame_gap > max_link_gap:
            continue

        clone = dict(hypothesis)
        # 오래 끌고만 있는 경로가 새 정상 후보보다 과하게 유리하지 않도록 작은 감점을 준다.
        clone["score"] = float(clone.get("score", 0.0)) - min(
            1.5,
            max(0, frame_gap - 1) * 0.08,
        )
        active.append(clone)
    return active


def _score_motion_transition(
    previous: Optional[Mapping[str, Any]],
    latest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target_frame: int,
    frame_key: str,
    x_key: str,
    y_key: str,
    confidence_key: str,
    max_distance_px: float,
    gap_growth_px: float,
    confidence_weight: float,
) -> Optional[Dict[str, float]]:
    """이전 경로에서 현재 후보로 자연스럽게 이어질 수 있는지 점수화한다."""
    latest_frame = _as_int(latest.get(frame_key))
    candidate_x = _as_float(candidate.get(x_key))
    candidate_y = _as_float(candidate.get(y_key))
    if latest_frame is None or candidate_x is None or candidate_y is None:
        return None

    frame_gap = target_frame - latest_frame
    if frame_gap <= 0:
        return None
    if frame_gap > 12:
        return None

    predicted_x, predicted_y = _predict_ball_candidate_position(
        previous,
        latest,
        target_frame,
        frame_key,
        x_key,
        y_key,
    )
    distance = math.hypot(candidate_x - predicted_x, candidate_y - predicted_y)

    dynamic_gate = max_distance_px + gap_growth_px * min(max(0, frame_gap - 1), 4)
    speed = _estimate_candidate_speed(previous, latest, frame_key, x_key, y_key)
    if speed is not None:
        dynamic_gate += min(max_distance_px * 0.45, speed * frame_gap * 0.22)
    dynamic_gate = min(
        dynamic_gate,
        max_distance_px + gap_growth_px * 4.0 + max_distance_px * 0.45,
    )
    if distance > dynamic_gate:
        return None

    quality = _motion_candidate_quality(candidate, confidence_key, x_key, y_key)
    confidence = _as_float(candidate.get(confidence_key)) or 0.0
    distance_penalty = distance / max(45.0, max_distance_px * 0.45)
    gap_penalty = max(0, frame_gap - 1) * 0.28
    acceleration_penalty = _candidate_acceleration_penalty(
        previous,
        latest,
        candidate,
        target_frame,
        frame_key,
        x_key,
        y_key,
    )

    score = (
        1.35
        + quality
        + confidence * max(0.0, confidence_weight) / 45.0
        - distance_penalty
        - gap_penalty
        - acceleration_penalty
    )
    return {"score": float(score), "distance": float(distance)}


def _candidate_acceleration_penalty(
    previous: Optional[Mapping[str, Any]],
    latest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target_frame: int,
    frame_key: str,
    x_key: str,
    y_key: str,
) -> float:
    """속도가 갑자기 바뀌는 후보를 감점해 순간이동처럼 보이는 점프를 줄인다."""
    if previous is None:
        return 0.0

    previous_x = _as_float(previous.get(x_key))
    previous_y = _as_float(previous.get(y_key))
    previous_frame = _as_int(previous.get(frame_key))
    latest_x = _as_float(latest.get(x_key))
    latest_y = _as_float(latest.get(y_key))
    latest_frame = _as_int(latest.get(frame_key))
    candidate_x = _as_float(candidate.get(x_key))
    candidate_y = _as_float(candidate.get(y_key))
    if None in (
        previous_x,
        previous_y,
        previous_frame,
        latest_x,
        latest_y,
        latest_frame,
        candidate_x,
        candidate_y,
    ):
        return 0.0

    previous_delta = float(int(latest_frame) - int(previous_frame))
    next_delta = float(target_frame - int(latest_frame))
    if previous_delta <= 0.0 or next_delta <= 0.0:
        return 0.0

    previous_vx = (float(latest_x) - float(previous_x)) / previous_delta
    previous_vy = (float(latest_y) - float(previous_y)) / previous_delta
    next_vx = (float(candidate_x) - float(latest_x)) / next_delta
    next_vy = (float(candidate_y) - float(latest_y)) / next_delta
    acceleration = math.hypot(next_vx - previous_vx, next_vy - previous_vy)
    return min(3.0, acceleration / 75.0)


def _make_motion_candidate_row(
    candidate: Mapping[str, Any],
    candidate_count: int,
    segment_id: int,
    association_distance: float,
    association_score: float,
    association_mode: str,
) -> Point:
    """선택된 공 후보에 추적 선택 메타데이터를 붙인다."""
    row = dict(candidate)
    row["candidate_count"] = int(candidate_count)
    row["association_distance"] = round(float(association_distance), 2)
    row["association_score"] = round(float(association_score), 2)
    row["association_mode"] = association_mode
    row["ball_segment"] = int(segment_id)
    return row


def _rank_motion_hypothesis(hypothesis: Mapping[str, Any]) -> float:
    """beam search에서 남길 경로의 우선순위를 계산한다."""
    return (
        float(hypothesis.get("score", 0.0))
        + int(hypothesis.get("hits", 0)) * 1.15
        - int(hypothesis.get("restarts", 0)) * 0.55
    )


__all__ = [
    "BallTracker",
    "interpolate_ball_path",
    "track_ball_detections",
    "correct_detections_with_ball_tracking",
    "correct_detection_csv",
]


if __name__ == "__main__":
    # Wrapper example after video_yolo.py creates video_detection.csv:
    #
    #   py ball_tracker.py
    #
    # Or inside video_yolo.py, after `df = pd.DataFrame(data)`:
    #
    #   from ball_tracker import correct_detections_with_ball_tracking
    #
    #   corrected_df = correct_detections_with_ball_tracking(
    #       df,
    #       total_frames=frame_id,
    #       max_gap=18,
    #       interpolation="auto",
    #       gravity_px_per_frame2=None,
    #       auto_gravity=True,
    #       enable_player_occlusion=True,
    #       player_center_radius_px=90,
    #   )
    #   corrected_df.to_csv("video_detection_ball_tracked.csv", index=False, encoding="utf-8-sig")
    #
    correct_detection_csv(
        input_csv="video_detection.csv",
        output_csv="video_detection_ball_tracked.csv",
        max_gap=18,
        interpolation="auto",
        gravity_px_per_frame2=None,
        auto_gravity=True,
    )
