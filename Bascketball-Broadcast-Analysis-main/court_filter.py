"""코트 바깥 객체를 제거하는 후처리 모듈.

YOLO가 관중석, 벤치, 광고판 주변 객체를 같이 검출할 수 있으므로,
코트 영역을 polygon으로 정의하고 그 안에 있는 객체만 남긴다.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Point = Tuple[float, float]
Detection = Dict[str, Any]


# 팀원용 역할 요약:
# 이 파일은 검출된 객체가 농구 코트 영역 안에 있는지 판단합니다.
# 관중석/벤치/광고판 쪽 객체가 CSV에 섞이면 선수·공 분석이 흔들리므로,
# normalized polygon으로 코트 영역을 정의하고 클래스별 anchor point를 기준으로 필터링합니다.
# 코트 모양이 영상마다 다르면 DEFAULT_BROADCAST_COURT_POLYGON_NORM 값을 조정하면 됩니다.


# 기본 코트 polygon이다. 좌표는 (x / frame_width, y / frame_height) 형태의 정규화 좌표다.
# 실제 영상 카메라 구도에 따라 이 값을 직접 조정하면 필터 정확도가 높아진다.
DEFAULT_BROADCAST_COURT_POLYGON_NORM: List[Point] = [
    (0.05, 0.30),
    (0.95, 0.30),
    (1.00, 0.98),
    (0.00, 0.98),
]

DEFAULT_ANCHOR_BY_CLASS = {
    # 사람은 중심점보다 발 위치가 코트 안/밖 판단에 더 적합하다.
    "person": "bottom_center",
    "player": "bottom_center",
    "referee": "bottom_center",
    # 공은 크기가 작으므로 중심점을 기준으로 판단한다.
    "sports ball": "center",
    "frisbee": "center",
    "basketball": "center",
    "ball": "center",
    # 골대는 코트 가장자리에 있으므로 중심점을 기준으로 남긴다.
    "hoop": "center",
    "rim": "center",
    "basketball hoop": "center",
}

DEFAULT_MARGIN_BY_CLASS = {
    # 공은 공중 이동 중 코트 바닥보다 훨씬 위에 떠 있을 수 있어 더 넓게 허용한다.
    "ball": 260.0,
    "sports ball": 260.0,
    "basketball": 260.0,
    "frisbee": 260.0,
}


class CourtObjectFilter:
    """농구 코트 polygon을 기준으로 YOLO 탐지 row를 필터링한다."""

    def __init__(
        self,
        court_polygon: Sequence[Any],
        frame_size: Optional[Tuple[int, int]] = None,
        polygon_is_normalized: bool = True,
        margin_px: float = 8.0,
        margin_by_class: Optional[Mapping[str, float]] = None,
        anchor_by_class: Optional[Mapping[str, str]] = None,
        default_anchor: str = "center",
    ) -> None:
        # 클래스별 기준점(anchor)을 기본값 위에 사용자 설정으로 덮어쓴다.
        # 클래스마다 코트 안/밖을 판단하는 기준점이 다릅니다.
        # 선수/심판은 발 위치(bottom_center), 공/골대는 중심점(center)을 사용합니다.
        self.frame_size = frame_size
        self.polygon_is_normalized = polygon_is_normalized
        self.margin_px = float(margin_px)
        self.margin_by_class = dict(DEFAULT_MARGIN_BY_CLASS)
        if margin_by_class:
            self.margin_by_class.update(
                {str(key): float(value) for key, value in margin_by_class.items()}
            )
        self.anchor_by_class = dict(DEFAULT_ANCHOR_BY_CLASS)
        if anchor_by_class:
            self.anchor_by_class.update(anchor_by_class)
        self.default_anchor = default_anchor
        self.court_polygon = normalize_polygon(
            court_polygon,
            frame_size=frame_size,
            polygon_is_normalized=polygon_is_normalized,
        )

    def keep_detection(
        self,
        detection: Mapping[str, Any],
        class_key: str = "class",
        x_key: str = "x_center",
        y_key: str = "y_center",
    ) -> bool:
        """탐지 객체 1개가 코트 안에 있으면 True를 반환한다."""
        class_name = str(detection.get(class_key))
        anchor = self.anchor_by_class.get(class_name, self.default_anchor)
        point = detection_anchor_point(detection, anchor=anchor, x_key=x_key, y_key=y_key)
        if point is None:
            return False
        margin = self.margin_by_class.get(class_name, self.margin_px)
        return point_in_polygon(point, self.court_polygon, margin_px=margin)

    def filter_records(
        self,
        detections: Iterable[Mapping[str, Any]],
        class_key: str = "class",
        x_key: str = "x_center",
        y_key: str = "y_center",
    ) -> List[Detection]:
        """여러 탐지 row 중 코트 안에 있는 row만 남긴다."""
        kept: List[Detection] = []
        for detection in detections:
            if self.keep_detection(detection, class_key=class_key, x_key=x_key, y_key=y_key):
                row = dict(detection)
                row["inside_court"] = True
                kept.append(row)
        return kept


def filter_detections_by_court(
    detections_df: Any,
    court_polygon: Optional[Sequence[Any]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    video_path: Optional[str] = None,
    polygon_is_normalized: bool = True,
    margin_px: float = 8.0,
    margin_by_class: Optional[Mapping[str, float]] = None,
    anchor_by_class: Optional[Mapping[str, str]] = None,
    default_anchor: str = "center",
    class_key: str = "class",
    x_key: str = "x_center",
    y_key: str = "y_center",
) -> Any:
    """DataFrame에서 코트 안 객체만 남긴 새 DataFrame을 반환한다."""

    import pandas as pd

    # polygon을 넘기지 않으면 기본 중계 화면용 코트 영역을 사용한다.
    # 기본 polygon은 방송 중계 화면에서 코트가 차지하는 대략적인 영역입니다.
    # 특정 영상에서 필터가 너무 넓거나 좁으면 court_polygon 또는 court_config_path로 조정합니다.
    polygon = DEFAULT_BROADCAST_COURT_POLYGON_NORM if court_polygon is None else court_polygon
    # 정규화 좌표를 실제 픽셀 좌표로 바꾸기 위해 영상 크기를 가져온다.
    size = resolve_frame_size(frame_size=frame_size, video_path=video_path)

    court_filter = CourtObjectFilter(
        polygon,
        frame_size=size,
        polygon_is_normalized=polygon_is_normalized,
        margin_px=margin_px,
        margin_by_class=margin_by_class,
        anchor_by_class=anchor_by_class,
        default_anchor=default_anchor,
    )

    kept_records = court_filter.filter_records(
        detections_df.to_dict("records"),
        class_key=class_key,
        x_key=x_key,
        y_key=y_key,
    )

    filtered_df = pd.DataFrame(kept_records)
    if filtered_df.empty:
        return detections_df.iloc[0:0].copy()

    # 후속 분석이 프레임 순서대로 처리되도록 정렬한다.
    sort_columns = [
        column
        for column in ["frame", class_key, "track_id"]
        if column in filtered_df.columns
    ]
    if sort_columns:
        filtered_df = filtered_df.sort_values(sort_columns).reset_index(drop=True)
    return filtered_df


def filter_detection_csv(
    input_csv: str = "video_detection.csv",
    output_csv: str = "video_detection_court_filtered.csv",
    court_polygon: Optional[Sequence[Any]] = None,
    court_config_path: Optional[str] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    video_path: Optional[str] = "Video Project.mp4",
    polygon_is_normalized: bool = True,
    margin_px: float = 8.0,
    margin_by_class: Optional[Mapping[str, float]] = None,
    **kwargs: Any,
) -> Any:
    """video_yolo.py를 수정하지 않고 CSV 파일만 후처리하는 wrapper."""

    import pandas as pd

    # JSON 설정 파일이 있으면 court_polygon, frame_size, margin 값을 거기서 읽는다.
    polygon = court_polygon
    config_normalized = polygon_is_normalized
    config_frame_size = frame_size
    config_margin = margin_px

    if court_config_path:
        config = load_court_config(court_config_path)
        polygon = config["polygon"]
        config_normalized = bool(config.get("normalized", polygon_is_normalized))
        config_frame_size = _tuple_frame_size(config.get("frame_size")) or frame_size
        config_margin = float(config.get("margin_px", margin_px))

    detections_df = pd.read_csv(input_csv)
    filtered_df = filter_detections_by_court(
        detections_df,
        court_polygon=polygon,
        frame_size=config_frame_size,
        video_path=video_path,
        polygon_is_normalized=config_normalized,
        margin_px=config_margin,
        margin_by_class=margin_by_class,
        **kwargs,
    )
    filtered_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return filtered_df


def load_court_config(path: str) -> Dict[str, Any]:
    """코트 polygon 설정 JSON 파일을 읽는다."""
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if isinstance(config, list):
        return {"polygon": config, "normalized": True}
    if not isinstance(config, dict) or "polygon" not in config:
        raise ValueError(
            "Court config must be a polygon list or a JSON object with a 'polygon' key."
        )
    return config


def resolve_frame_size(
    frame_size: Optional[Tuple[int, int]] = None,
    video_path: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """직접 받은 frame_size를 우선 사용하고, 없으면 video_path에서 영상 크기를 읽는다."""
    if frame_size is not None:
        return _tuple_frame_size(frame_size)
    if video_path:
        return read_video_frame_size(video_path)
    return None


def read_video_frame_size(video_path: str) -> Optional[Tuple[int, int]]:
    """OpenCV로 영상의 width/height를 읽는다."""
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        return None
    return width, height


def normalize_polygon(
    polygon: Sequence[Any],
    frame_size: Optional[Tuple[int, int]],
    polygon_is_normalized: bool,
) -> List[Point]:
    """정규화 polygon 좌표를 실제 픽셀 좌표로 변환한다."""
    points = [_parse_point(point) for point in polygon]
    if len(points) < 3:
        raise ValueError("court_polygon must contain at least 3 points.")

    if not polygon_is_normalized:
        return points

    if frame_size is None:
        raise ValueError(
            "frame_size or video_path is required when court_polygon uses normalized coordinates."
        )

    width, height = frame_size
    return [(x * width, y * height) for x, y in points]


def detection_anchor_point(
    detection: Mapping[str, Any],
    anchor: str,
    x_key: str = "x_center",
    y_key: str = "y_center",
) -> Optional[Point]:
    """클래스별 기준점(anchor)에 맞춰 코트 판정용 좌표를 만든다."""
    x_center = _as_float(detection.get(x_key))
    y_center = _as_float(detection.get(y_key))
    if x_center is None or y_center is None:
        return None

    x1 = _as_float(detection.get("x1"))
    y1 = _as_float(detection.get("y1"))
    x2 = _as_float(detection.get("x2"))
    y2 = _as_float(detection.get("y2"))

    if anchor == "center":
        return x_center, y_center
    if anchor == "bottom_center" and y2 is not None:
        return x_center, y2
    if anchor == "top_center" and y1 is not None:
        return x_center, y1
    if anchor == "bbox_center" and None not in (x1, y1, x2, y2):
        return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
    return x_center, y_center


def point_in_polygon(
    point: Point,
    polygon: Sequence[Point],
    margin_px: float = 0.0,
) -> bool:
    """점이 polygon 안에 있는지 ray casting 방식으로 판정한다."""
    if margin_px > 0.0 and _distance_to_polygon(point, polygon) <= margin_px:
        return True

    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]

        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True

        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _distance_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    """점과 polygon 테두리 사이의 최단 거리를 계산한다."""
    return min(
        _distance_to_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _distance_to_segment(
    point: Point,
    segment_start: Point,
    segment_end: Point,
) -> float:
    """점과 선분 사이의 최단 거리를 계산한다."""
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _point_on_segment(
    point: Point,
    segment_start: Point,
    segment_end: Point,
    eps: float = 1e-7,
) -> bool:
    """점이 선분 위에 거의 붙어 있는지 확인한다."""
    return _distance_to_segment(point, segment_start, segment_end) <= eps


def _parse_point(point: Any) -> Point:
    """list/tuple 또는 {'x': ..., 'y': ...} 형태의 좌표를 Point로 변환한다."""
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    if isinstance(point, Sequence) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise ValueError(f"Invalid polygon point: {point!r}")


def _tuple_frame_size(value: Any) -> Optional[Tuple[int, int]]:
    """frame_size 값을 (width, height) 튜플로 변환한다."""
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) >= 2:
        return int(value[0]), int(value[1])
    raise ValueError("frame_size must be a tuple/list like (width, height).")


def _as_float(value: Any) -> Optional[float]:
    """값을 float로 안전하게 변환한다. NaN/inf는 None으로 처리한다."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


__all__ = [
    "CourtObjectFilter",
    "DEFAULT_BROADCAST_COURT_POLYGON_NORM",
    "DEFAULT_MARGIN_BY_CLASS",
    "filter_detections_by_court",
    "filter_detection_csv",
    "load_court_config",
    "point_in_polygon",
]


if __name__ == "__main__":
    # After video_yolo.py creates video_detection.csv:
    #
    #   py court_filter.py
    #
    # For better accuracy, tune the polygon for your camera view:
    #
    #   filtered_df = filter_detection_csv(
    #       input_csv="video_detection.csv",
    #       output_csv="video_detection_court_filtered.csv",
    #       video_path="Video Project.mp4",
    #       court_polygon=[
    #           (0.05, 0.30),
    #           (0.95, 0.30),
    #           (1.00, 0.98),
    #           (0.00, 0.98),
    #       ],
    #       polygon_is_normalized=True,
    #   )
    #
    filter_detection_csv()
