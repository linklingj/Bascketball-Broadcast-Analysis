"""농구 추적 결과에 팀 정보와 이벤트 측정값을 추가합니다.

현재 객체탐지 모델은 선수를 팀별 클래스가 아닌 일반 player로 구분합니다.
그래서 선수 유니폼 색상을 이용해 team_1, team_2를 추정하고, 정리된 객체
좌표를 바탕으로 슛, 패스, 스틸/블락, 파울, 리바운드, 볼점유율을 계산합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


PLAYER_CLASSES = {"player", "person"}
BALL_CLASSES = {"ball", "sports ball", "basketball", "frisbee"}
HOOP_CLASSES = {"hoop", "rim", "basket", "basketball hoop"}

TEAM_COLUMNS = ["team_id", "team_name", "team_confidence", "team_color_hex"]

EVENT_ORDER = [
    "shot_attempt",
    "shot_made",
    "shot_missed",
    "pass",
    "steal_or_block",
    "foul",
    "rebound",
]

EVENT_COUNT_COLUMNS = [
    f"{event_type}_count" for event_type in EVENT_ORDER
] + [
    "shot_success_rate_pct",
    "possession_frame_count",
    "possession_time_sec",
    "possession_pct",
]

EVENT_LABELS_KO = {
    "shot_attempt": "슛시도",
    "shot_made": "슛성공",
    "shot_missed": "슛실패",
    "pass": "패스",
    "steal_or_block": "스틸/블락",
    "foul": "파울",
    "rebound": "리바운드",
}

EVENT_COLUMNS = [
    "row_type",
    "event_id",
    "event_type",
    "event_label_ko",
    "count",
    "frame",
    "time_sec",
    "team_id",
    "team_name",
    *EVENT_COUNT_COLUMNS,
    "player_track_id",
    "secondary_track_id",
    "confidence",
    "details",
]


@dataclass
class TeamAssignment:
    team_id: int
    team_name: str
    team_confidence: float
    team_color_hex: str


@dataclass
class PossessionSegment:
    start_frame: int
    end_frame: int
    player_track_id: int
    team_id: str
    team_name: str
    raw_frame_count: int


@dataclass
class ShotWindow:
    start_frame: int
    end_frame: int
    event_frame: int
    made: bool
    confidence: float
    team_id: str
    team_name: str
    player_track_id: Optional[int]
    min_rim_distance: float


def assign_player_teams_from_video(
    tracking_df: pd.DataFrame,
    video_path: str,
    frame_size: Tuple[int, int],
    start_frame: int = 0,
    sample_stride_frames: int = 12,
    max_samples_per_track: int = 48,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign team ids to player rows using uniform color samples.

    The model classes are not team-specific, so this pass samples the upper body
    area from each tracked player box and clusters player tracks into two color
    groups. Non-player rows keep empty team columns.
    """

    output_df = _ensure_team_columns(tracking_df.copy())
    summary: Dict[str, Any] = {
        "method": "uniform_color_clustering",
        "team_class_source": "color_clustering",
        "tracks_sampled": 0,
        "teams_found": 0,
        "note": "",
    }

    if output_df.empty or "class" not in output_df.columns:
        summary["note"] = "No rows available for team assignment."
        return output_df, summary

    numeric_df = _coerce_tracking_numbers(output_df)
    player_df = numeric_df[
        numeric_df["class"].astype(str).str.lower().isin(PLAYER_CLASSES)
        & numeric_df["track_id"].notna()
        & numeric_df["frame"].notna()
    ].copy()
    player_df = player_df[player_df["track_id"] >= 0]
    if player_df.empty:
        summary["note"] = "No tracked player rows were available."
        return output_df, summary

    track_features = _sample_player_track_colors(
        player_df=player_df,
        video_path=video_path,
        frame_size=frame_size,
        start_frame=start_frame,
        sample_stride_frames=sample_stride_frames,
        max_samples_per_track=max_samples_per_track,
    )
    if len(track_features) < 2 and sample_stride_frames > 1:
        track_features = _sample_player_track_colors(
            player_df=player_df,
            video_path=video_path,
            frame_size=frame_size,
            start_frame=start_frame,
            sample_stride_frames=1,
            max_samples_per_track=max_samples_per_track,
        )
    summary["tracks_sampled"] = len(track_features)
    if len(track_features) < 2:
        summary["note"] = "Not enough sampled player tracks to split two teams."
        return output_df, summary

    assignments, team_summary = _cluster_track_features(track_features)
    summary.update(team_summary)
    if not assignments:
        summary["note"] = "Player colors were too similar to split teams reliably."
        return output_df, summary

    _apply_team_assignments(output_df, assignments)
    return output_df, summary


def assign_fallback_player_track_ids(
    tracking_df: pd.DataFrame,
    frame_size: Tuple[int, int],
    max_gap_frames: int = 12,
    max_match_distance_px: Optional[float] = None,
) -> pd.DataFrame:
    """Add synthetic player track ids when the tracker leaves players at -1."""

    output_df = _coerce_tracking_numbers(tracking_df.copy())
    if output_df.empty or "class" not in output_df.columns:
        return tracking_df

    player_mask = output_df["class"].astype(str).str.lower().isin(PLAYER_CLASSES)
    missing_mask = player_mask & (
        output_df["track_id"].isna() | (output_df["track_id"] < 0)
    )
    if not bool(missing_mask.any()):
        return tracking_df

    existing_ids = output_df.loc[player_mask & output_df["track_id"].notna() & (output_df["track_id"] >= 0), "track_id"]
    next_track_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
    scale = max(0.75, min(1.6, max(frame_size) / 1280.0))
    max_distance = float(max_match_distance_px) if max_match_distance_px is not None else 175.0 * scale
    max_gap = max(1, int(max_gap_frames))

    active_tracks: Dict[int, Dict[str, Any]] = {}
    for frame in sorted(output_df.loc[player_mask, "frame"].dropna().astype(int).unique()):
        frame_indices = output_df.index[player_mask & output_df["frame"].eq(frame)].tolist()
        missing_indices = [
            index
            for index in frame_indices
            if pd.isna(output_df.at[index, "track_id"]) or float(output_df.at[index, "track_id"]) < 0
        ]

        stale_ids = [
            track_id
            for track_id, state in active_tracks.items()
            if frame - int(state["frame"]) > max_gap
        ]
        for track_id in stale_ids:
            active_tracks.pop(track_id, None)

        candidates: List[Tuple[float, int, int]] = []
        for index in missing_indices:
            center = _center(output_df.loc[index].to_dict())
            if center is None:
                continue
            for track_id, state in active_tracks.items():
                distance = math.hypot(center[0] - state["center"][0], center[1] - state["center"][1])
                if distance <= max_distance:
                    candidates.append((distance, index, track_id))

        assigned_indices: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, index, track_id in sorted(candidates, key=lambda item: item[0]):
            if index in assigned_indices or track_id in assigned_tracks:
                continue
            output_df.at[index, "track_id"] = track_id
            center = _center(output_df.loc[index].to_dict())
            if center is not None:
                active_tracks[track_id] = {"center": center, "frame": frame}
            assigned_indices.add(index)
            assigned_tracks.add(track_id)

        for index in missing_indices:
            if index in assigned_indices:
                continue
            center = _center(output_df.loc[index].to_dict())
            if center is None:
                continue
            track_id = next_track_id
            next_track_id += 1
            output_df.at[index, "track_id"] = track_id
            active_tracks[track_id] = {"center": center, "frame": frame}

        for index in frame_indices:
            if index in missing_indices:
                continue
            try:
                track_id = int(float(output_df.at[index, "track_id"]))
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue
            center = _center(output_df.loc[index].to_dict())
            if center is not None:
                active_tracks[track_id] = {"center": center, "frame": frame}

    return output_df


def write_event_measurements(
    tracking_df: pd.DataFrame,
    output_csv_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    video_path: str = "",
) -> pd.DataFrame:
    """Estimate events and write one event-measurement CSV."""

    event_df = measure_basketball_events(
        tracking_df=tracking_df,
        fps=fps,
        frame_size=frame_size,
        video_path=video_path,
    )
    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(output, index=False, encoding="utf-8-sig")
    return event_df


def measure_basketball_events(
    tracking_df: pd.DataFrame,
    fps: float,
    frame_size: Tuple[int, int],
    video_path: str = "",
) -> pd.DataFrame:
    """Create summary and detail rows for basketball events.

    These measurements are heuristic because broadcast video object boxes do not
    include whistle audio, scoreboard state, player pose, or explicit ball
    ownership labels.
    """

    del video_path
    fps_value = float(fps) if fps and fps > 0 else 30.0
    df = _coerce_tracking_numbers(_ensure_team_columns(tracking_df.copy()))
    if df.empty:
        return _event_output_dataframe([], fps_value)

    known_teams = _known_teams_from_tracking(df)
    frame_state = _build_frame_state(df)
    possession_by_frame = _infer_frame_possessions(frame_state, frame_size)
    possession_segments = _build_possession_segments(possession_by_frame, fps_value)
    possession_stats = _build_possession_stats(possession_by_frame, fps_value)

    events: List[Dict[str, Any]] = []
    shot_windows = _detect_shot_windows(
        frame_state=frame_state,
        possession_segments=possession_segments,
        fps=fps_value,
        frame_size=frame_size,
    )
    for shot in shot_windows:
        actor = "" if shot.player_track_id is None else str(shot.player_track_id)
        events.append(
            _make_event_row(
                event_type="shot_attempt",
                frame=shot.event_frame,
                fps=fps_value,
                team_id=shot.team_id,
                team_name=shot.team_name,
                player_track_id=actor,
                confidence=shot.confidence,
                details=f"near_rim_distance_px={shot.min_rim_distance:.1f}",
            )
        )
        events.append(
            _make_event_row(
                event_type="shot_made" if shot.made else "shot_missed",
                frame=shot.event_frame,
                fps=fps_value,
                team_id=shot.team_id,
                team_name=shot.team_name,
                player_track_id=actor,
                confidence=max(0.25, shot.confidence - 0.12),
                details="ball_crossed_rim_zone" if shot.made else "near_rim_no_make_detected",
            )
        )

    events.extend(
        _detect_passes_and_steals(
            possession_segments=possession_segments,
            fps=fps_value,
            shot_windows=shot_windows,
        )
    )
    events.extend(
        _detect_rebounds(
            possession_segments=possession_segments,
            shot_windows=shot_windows,
            fps=fps_value,
        )
    )
    events.extend(
        _detect_contact_fouls(
            frame_state=frame_state,
            possession_segments=possession_segments,
            fps=fps_value,
            frame_size=frame_size,
            shot_windows=shot_windows,
        )
    )

    return _event_output_dataframe(
        events,
        fps_value,
        known_teams=known_teams,
        possession_stats=possession_stats,
    )


def _ensure_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in TEAM_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _coerce_tracking_numbers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "frame",
        "track_id",
        "confidence",
        "x_center",
        "y_center",
        "x1",
        "y1",
        "x2",
        "y2",
        "team_confidence",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _sample_player_track_colors(
    player_df: pd.DataFrame,
    video_path: str,
    frame_size: Tuple[int, int],
    start_frame: int,
    sample_stride_frames: int,
    max_samples_per_track: int,
) -> Dict[int, np.ndarray]:
    stride = max(1, int(sample_stride_frames))
    sample_df = player_df[player_df["frame"].astype(int) % stride == 0].copy()
    if sample_df.empty:
        sample_df = player_df.copy()

    rows_by_frame: Dict[int, List[Mapping[str, Any]]] = {}
    for row in sample_df.sort_values(["frame", "track_id"]).to_dict("records"):
        frame = int(row["frame"])
        rows_by_frame.setdefault(frame, []).append(row)

    if not rows_by_frame:
        return {}

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {}
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    max_relative_frame = max(rows_by_frame)
    samples: Dict[int, List[np.ndarray]] = {}
    frame_id = 0
    while capture.isOpened() and frame_id <= max_relative_frame:
        ret, frame = capture.read()
        if not ret:
            break

        for row in rows_by_frame.get(frame_id, []):
            track_id = int(row["track_id"])
            track_samples = samples.setdefault(track_id, [])
            if len(track_samples) >= max(1, int(max_samples_per_track)):
                continue

            feature = _player_uniform_feature(frame, row, frame_size)
            if feature is not None:
                track_samples.append(feature)

        frame_id += 1

    capture.release()

    track_features: Dict[int, np.ndarray] = {}
    for track_id, track_samples in samples.items():
        if not track_samples:
            continue
        stacked = np.vstack(track_samples)
        track_features[track_id] = np.median(stacked, axis=0)
    return track_features


def _player_uniform_feature(
    frame: Any,
    row: Mapping[str, Any],
    frame_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    bbox = _safe_bbox(row, frame_size)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width < 12 or height < 24:
        return None

    torso_x1 = x1 + int(width * 0.18)
    torso_x2 = x2 - int(width * 0.18)
    torso_y1 = y1 + int(height * 0.18)
    torso_y2 = y1 + int(height * 0.68)
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (value > 28) & (value < 248) & (saturation > 18)
    if int(mask.sum()) < 24:
        mask = value > 28
    if int(mask.sum()) < 24:
        return None

    pixels = lab[mask]
    median_lab = np.median(pixels.astype(float), axis=0)
    return median_lab


def _cluster_track_features(
    track_features: Mapping[int, np.ndarray],
) -> Tuple[Dict[int, TeamAssignment], Dict[str, Any]]:
    track_ids = sorted(track_features)
    features = np.vstack([track_features[track_id] for track_id in track_ids]).astype(float)
    if len(track_ids) < 2:
        return {}, {"teams_found": 0}

    distance_matrix = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distance_matrix)), distance_matrix.shape)
    max_distance = float(distance_matrix[first, second])
    if max_distance < 10.0:
        return {}, {"teams_found": 0, "team_color_distance": round(max_distance, 3)}

    centers = np.vstack([features[first], features[second]]).astype(float)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(30):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for label in (0, 1):
            members = features[labels == label]
            if len(members) > 0:
                centers[label] = np.mean(members, axis=0)

    # Stable public numbering: team 1 is the lighter uniform cluster.
    light_order = sorted((0, 1), key=lambda label: centers[label][0], reverse=True)
    label_to_team_id = {light_order[0]: 1, light_order[1]: 2}
    team_names = {1: "team_1", 2: "team_2"}
    team_colors = {
        team_id: _lab_to_hex(centers[label])
        for label, team_id in label_to_team_id.items()
    }

    assignments: Dict[int, TeamAssignment] = {}
    for index, track_id in enumerate(track_ids):
        label = int(labels[index])
        team_id = label_to_team_id[label]
        distances = np.linalg.norm(features[index][None, :] - centers, axis=1)
        best_distance = float(distances[label])
        other_distance = float(distances[1 - label])
        if other_distance <= 1e-6:
            confidence = 0.5
        else:
            margin = max(0.0, other_distance - best_distance) / other_distance
            confidence = 0.5 + 0.49 * min(1.0, margin)
        assignments[track_id] = TeamAssignment(
            team_id=team_id,
            team_name=team_names[team_id],
            team_confidence=round(confidence, 3),
            team_color_hex=team_colors[team_id],
        )

    return assignments, {
        "teams_found": 2,
        "team_color_distance": round(max_distance, 3),
        "team_1_color_hex": team_colors[1],
        "team_2_color_hex": team_colors[2],
    }


def _lab_to_hex(lab_color: Sequence[float]) -> str:
    lab_pixel = np.array([[np.clip(lab_color, 0, 255)]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"


def _apply_team_assignments(
    df: pd.DataFrame,
    assignments: Mapping[int, TeamAssignment],
) -> None:
    player_mask = df["class"].astype(str).str.lower().isin(PLAYER_CLASSES)
    for index, row in df[player_mask].iterrows():
        try:
            track_id = int(float(row.get("track_id")))
        except (TypeError, ValueError):
            continue
        assignment = assignments.get(track_id)
        if assignment is None:
            df.at[index, "team_name"] = "unknown"
            continue
        df.at[index, "team_id"] = str(assignment.team_id)
        df.at[index, "team_name"] = assignment.team_name
        df.at[index, "team_confidence"] = assignment.team_confidence
        df.at[index, "team_color_hex"] = assignment.team_color_hex


def _build_frame_state(df: pd.DataFrame) -> Dict[str, Dict[int, Any]]:
    balls: Dict[int, Mapping[str, Any]] = {}
    hoops: Dict[int, Mapping[str, Any]] = {}
    players: Dict[int, List[Mapping[str, Any]]] = {}

    if df.empty:
        return {"balls": balls, "hoops": hoops, "players": players}

    for frame, frame_df in df.dropna(subset=["frame"]).groupby("frame", sort=True):
        frame_id = int(frame)
        rows = frame_df.to_dict("records")
        ball_rows = [row for row in rows if str(row.get("class", "")).lower() in BALL_CLASSES]
        hoop_rows = [row for row in rows if str(row.get("class", "")).lower() in HOOP_CLASSES]
        player_rows = [row for row in rows if str(row.get("class", "")).lower() in PLAYER_CLASSES]
        if ball_rows:
            balls[frame_id] = max(ball_rows, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        if hoop_rows:
            hoops[frame_id] = max(hoop_rows, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        if player_rows:
            players[frame_id] = player_rows

    return {"balls": balls, "hoops": hoops, "players": players}


def _infer_frame_possessions(
    frame_state: Mapping[str, Dict[int, Any]],
    frame_size: Tuple[int, int],
) -> Dict[int, Mapping[str, Any]]:
    scale = max(0.75, min(1.6, max(frame_size) / 1280.0))
    possessions: Dict[int, Mapping[str, Any]] = {}
    balls = frame_state["balls"]
    players_by_frame = frame_state["players"]

    for frame, ball in balls.items():
        players = players_by_frame.get(frame, [])
        possessor = _nearest_possessor(ball, players, scale)
        if possessor is None:
            continue
        try:
            player_track_id = int(float(possessor.get("track_id")))
        except (TypeError, ValueError):
            continue
        possessions[frame] = {
            "frame": frame,
            "player_track_id": player_track_id,
            "team_id": _clean_team_id(possessor.get("team_id")),
            "team_name": _clean_text(possessor.get("team_name")),
        }

    return possessions


def _nearest_possessor(
    ball: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    scale: float,
) -> Optional[Mapping[str, Any]]:
    ball_point = _center(ball)
    if ball_point is None:
        return None

    best_player: Optional[Mapping[str, Any]] = None
    best_distance = float("inf")
    for player in players:
        try:
            x1, y1, x2, y2 = float(player["x1"]), float(player["y1"]), float(player["x2"]), float(player["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        distance = _point_rect_distance(ball_point[0], ball_point[1], x1, y1, x2, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        threshold = max(55.0 * scale, min(150.0 * scale, height * 0.32 + width * 0.22))
        if distance <= threshold and distance < best_distance:
            best_distance = distance
            best_player = player

    return best_player


def _build_possession_segments(
    possession_by_frame: Mapping[int, Mapping[str, Any]],
    fps: float,
) -> List[PossessionSegment]:
    if not possession_by_frame:
        return []

    max_gap = max(2, int(round(fps * 0.35)))
    min_raw_frames = max(1, int(round(fps * 0.06)))
    segments: List[PossessionSegment] = []
    current: Optional[PossessionSegment] = None

    for frame in sorted(possession_by_frame):
        item = possession_by_frame[frame]
        player_track_id = int(item["player_track_id"])
        team_id = str(item.get("team_id", ""))
        team_name = str(item.get("team_name", ""))

        if (
            current is not None
            and current.player_track_id == player_track_id
            and frame - current.end_frame <= max_gap
        ):
            current.end_frame = frame
            current.raw_frame_count += 1
            if not current.team_id and team_id:
                current.team_id = team_id
                current.team_name = team_name
            continue

        if current is not None and current.raw_frame_count >= min_raw_frames:
            segments.append(current)
        current = PossessionSegment(
            start_frame=frame,
            end_frame=frame,
            player_track_id=player_track_id,
            team_id=team_id,
            team_name=team_name,
            raw_frame_count=1,
        )

    if current is not None and current.raw_frame_count >= min_raw_frames:
        segments.append(current)
    return segments


def _build_possession_stats(
    possession_by_frame: Mapping[int, Mapping[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    """Aggregate inferred ball possession frames by team."""

    fps_value = float(fps) if fps and fps > 0 else 30.0
    team_frames: Dict[str, int] = {}
    team_names: Dict[str, str] = {}
    for item in possession_by_frame.values():
        team_id = _clean_team_id(item.get("team_id"))
        if not team_id:
            continue
        team_frames[team_id] = team_frames.get(team_id, 0) + 1
        team_names.setdefault(team_id, _clean_text(item.get("team_name")) or f"team_{team_id}")

    total_frames = sum(team_frames.values())
    teams: Dict[str, Dict[str, Any]] = {}
    for team_id, frame_count in team_frames.items():
        teams[team_id] = {
            "team_id": team_id,
            "team_name": team_names.get(team_id, f"team_{team_id}"),
            "possession_frame_count": int(frame_count),
            "possession_time_sec": round(frame_count / fps_value, 3),
            "possession_pct": round((frame_count / total_frames) * 100.0, 2)
            if total_frames > 0
            else 0.0,
        }

    return {
        "total_possession_frames": int(total_frames),
        "total_possession_time_sec": round(total_frames / fps_value, 3),
        "teams": teams,
    }


def _detect_shot_windows(
    frame_state: Mapping[str, Dict[int, Any]],
    possession_segments: Sequence[PossessionSegment],
    fps: float,
    frame_size: Tuple[int, int],
) -> List[ShotWindow]:
    balls = frame_state["balls"]
    hoops = frame_state["hoops"]
    if not balls or not hoops:
        return []

    scale = max(0.75, min(1.6, max(frame_size) / 1280.0))
    near_frames: List[Tuple[int, float]] = []
    for frame, ball in balls.items():
        hoop = hoops.get(frame)
        if hoop is None:
            continue
        ball_center = _center(ball)
        hoop_center = _center(hoop)
        if ball_center is None or hoop_center is None:
            continue
        try:
            hoop_w = abs(float(hoop["x2"]) - float(hoop["x1"]))
            hoop_h = abs(float(hoop["y2"]) - float(hoop["y1"]))
        except (KeyError, TypeError, ValueError):
            hoop_w, hoop_h = 60.0 * scale, 40.0 * scale
        rim_radius = max(75.0 * scale, min(230.0 * scale, max(hoop_w * 1.8, hoop_h * 2.4)))
        distance = math.hypot(ball_center[0] - hoop_center[0], ball_center[1] - hoop_center[1])
        if distance <= rim_radius:
            near_frames.append((frame, distance))

    if not near_frames:
        return []

    clusters = _cluster_frame_runs(near_frames, max_gap=max(3, int(round(fps * 0.55))))
    min_separation = max(10, int(round(fps * 1.2)))
    shot_windows: List[ShotWindow] = []
    last_event_frame = -min_separation
    for cluster in clusters:
        if not cluster:
            continue
        event_frame, min_distance = min(cluster, key=lambda item: item[1])
        if event_frame - last_event_frame < min_separation:
            continue

        start_frame = cluster[0][0]
        end_frame = cluster[-1][0]
        possessor = _find_last_possession_before(
            possession_segments,
            frame=event_frame,
            max_lookback_frames=max(12, int(round(fps * 4.0))),
        )
        made = _shot_cluster_made(frame_state, cluster, fps)
        confidence = _shot_confidence(min_distance, cluster)
        shot_windows.append(
            ShotWindow(
                start_frame=start_frame,
                end_frame=end_frame,
                event_frame=event_frame,
                made=made,
                confidence=confidence,
                team_id=possessor.team_id if possessor else "",
                team_name=possessor.team_name if possessor else "",
                player_track_id=possessor.player_track_id if possessor else None,
                min_rim_distance=float(min_distance),
            )
        )
        last_event_frame = event_frame

    return shot_windows


def _cluster_frame_runs(items: Sequence[Tuple[int, float]], max_gap: int) -> List[List[Tuple[int, float]]]:
    clusters: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = []
    for frame, value in sorted(items):
        if current and frame - current[-1][0] > max_gap:
            clusters.append(current)
            current = []
        current.append((frame, value))
    if current:
        clusters.append(current)
    return clusters


def _shot_cluster_made(
    frame_state: Mapping[str, Dict[int, Any]],
    cluster: Sequence[Tuple[int, float]],
    fps: float,
) -> bool:
    balls = frame_state["balls"]
    hoops = frame_state["hoops"]
    if not cluster:
        return False

    start = cluster[0][0]
    end = cluster[-1][0] + max(3, int(round(fps * 0.75)))
    candidate_frames = [frame for frame in sorted(balls) if start <= frame <= end and frame in hoops]
    previous_y: Optional[float] = None
    previous_frame_inside_x = False
    for frame in candidate_frames:
        ball = balls[frame]
        hoop = hoops[frame]
        ball_center = _center(ball)
        if ball_center is None:
            continue
        try:
            x1, y1, x2, y2 = float(hoop["x1"]), float(hoop["y1"]), float(hoop["x2"]), float(hoop["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        hoop_w = max(8.0, abs(x2 - x1))
        hoop_h = max(8.0, abs(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        inside_x = (x1 - hoop_w * 0.45) <= ball_center[0] <= (x2 + hoop_w * 0.45)
        in_score_zone = inside_x and (y1 - hoop_h * 0.35) <= ball_center[1] <= (y2 + hoop_h * 1.45)
        crossed_down = (
            previous_y is not None
            and previous_frame_inside_x
            and previous_y < cy <= ball_center[1]
        )
        if in_score_zone and (ball_center[1] >= cy or crossed_down):
            return True
        previous_y = ball_center[1]
        previous_frame_inside_x = inside_x
    return False


def _shot_confidence(min_distance: float, cluster: Sequence[Tuple[int, float]]) -> float:
    duration_bonus = min(0.18, len(cluster) * 0.015)
    proximity_bonus = max(0.0, min(0.22, (90.0 - min_distance) / 260.0))
    return round(max(0.32, min(0.82, 0.42 + duration_bonus + proximity_bonus)), 3)


def _detect_passes_and_steals(
    possession_segments: Sequence[PossessionSegment],
    fps: float,
    shot_windows: Sequence[ShotWindow],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    max_transition_gap = max(6, int(round(fps * 3.0)))
    for left, right in zip(possession_segments, possession_segments[1:]):
        if left.player_track_id == right.player_track_id:
            continue
        gap = right.start_frame - left.end_frame
        if gap < 0 or gap > max_transition_gap:
            continue
        if _near_made_shot(right.start_frame, shot_windows, fps):
            continue

        same_known_team = left.team_id and right.team_id and left.team_id == right.team_id
        different_known_team = left.team_id and right.team_id and left.team_id != right.team_id
        if same_known_team:
            events.append(
                _make_event_row(
                    event_type="pass",
                    frame=right.start_frame,
                    fps=fps,
                    team_id=right.team_id,
                    team_name=right.team_name,
                    player_track_id=str(right.player_track_id),
                    secondary_track_id=str(left.player_track_id),
                    confidence=0.58,
                    details=f"possession_changed_same_team;gap_frames={gap}",
                )
            )
        elif different_known_team:
            detail = "possession_changed_opponent_team"
            if _near_missed_shot(right.start_frame, shot_windows, fps):
                detail = "opponent_possession_after_missed_shot_or_block"
            events.append(
                _make_event_row(
                    event_type="steal_or_block",
                    frame=right.start_frame,
                    fps=fps,
                    team_id=right.team_id,
                    team_name=right.team_name,
                    player_track_id=str(right.player_track_id),
                    secondary_track_id=str(left.player_track_id),
                    confidence=0.52,
                    details=f"{detail};gap_frames={gap}",
                )
            )
    return events


def _detect_rebounds(
    possession_segments: Sequence[PossessionSegment],
    shot_windows: Sequence[ShotWindow],
    fps: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    max_lookahead = max(8, int(round(fps * 5.0)))
    for shot in shot_windows:
        if shot.made:
            continue
        rebound = _find_first_possession_after(
            possession_segments,
            frame=shot.end_frame,
            max_lookahead_frames=max_lookahead,
        )
        if rebound is None:
            continue
        kind = "offensive" if shot.team_id and rebound.team_id == shot.team_id else "defensive"
        events.append(
            _make_event_row(
                event_type="rebound",
                frame=rebound.start_frame,
                fps=fps,
                team_id=rebound.team_id,
                team_name=rebound.team_name,
                player_track_id=str(rebound.player_track_id),
                confidence=0.5,
                details=f"{kind}_rebound_after_missed_shot_frame={shot.event_frame}",
            )
        )
    return events


def _detect_contact_fouls(
    frame_state: Mapping[str, Dict[int, Any]],
    possession_segments: Sequence[PossessionSegment],
    fps: float,
    frame_size: Tuple[int, int],
    shot_windows: Sequence[ShotWindow],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    cooldown = max(10, int(round(fps * 2.5)))
    last_foul_frame = -cooldown
    for left, right in zip(possession_segments, possession_segments[1:]):
        if not left.team_id or not right.team_id or left.team_id == right.team_id:
            continue
        frame = right.start_frame
        if frame - last_foul_frame < cooldown or _frame_in_any_shot_window(frame, shot_windows, fps):
            continue
        contact = _opponent_contact_near_ball(frame_state, frame, fps, frame_size)
        if contact is None:
            continue
        primary, secondary = contact
        events.append(
            _make_event_row(
                event_type="foul",
                frame=frame,
                fps=fps,
                team_id=right.team_id,
                team_name=right.team_name,
                player_track_id=str(primary),
                secondary_track_id=str(secondary),
                confidence=0.28,
                details="heuristic_contact_near_ball_during_opponent_possession_change",
            )
        )
        last_foul_frame = frame
    return events


def _opponent_contact_near_ball(
    frame_state: Mapping[str, Dict[int, Any]],
    frame: int,
    fps: float,
    frame_size: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    scale = max(0.75, min(1.6, max(frame_size) / 1280.0))
    search_radius = max(2, int(round(fps * 0.35)))
    close_player_distance = 85.0 * scale
    ball_distance = 130.0 * scale
    balls = frame_state["balls"]
    players_by_frame = frame_state["players"]

    for near_frame in range(frame - search_radius, frame + search_radius + 1):
        ball = balls.get(near_frame)
        players = players_by_frame.get(near_frame, [])
        ball_center = _center(ball) if ball is not None else None
        if ball_center is None or len(players) < 2:
            continue
        for index, left in enumerate(players):
            left_team = _clean_team_id(left.get("team_id"))
            if not left_team:
                continue
            for right in players[index + 1 :]:
                right_team = _clean_team_id(right.get("team_id"))
                if not right_team or left_team == right_team:
                    continue
                if not _players_are_close(left, right, close_player_distance):
                    continue
                left_dist = _point_to_record_bbox_distance(ball_center, left)
                right_dist = _point_to_record_bbox_distance(ball_center, right)
                if min(left_dist, right_dist) > ball_distance:
                    continue
                try:
                    return int(float(left["track_id"])), int(float(right["track_id"]))
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def _players_are_close(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    center_distance_threshold: float,
) -> bool:
    left_center = _center(left)
    right_center = _center(right)
    if left_center is None or right_center is None:
        return False
    center_distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    return center_distance <= center_distance_threshold or _bbox_iou(left, right) >= 0.04


def _bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    try:
        lx1, ly1, lx2, ly2 = float(left["x1"]), float(left["y1"]), float(left["x2"]), float(left["y2"])
        rx1, ry1, rx2, ry2 = float(right["x1"]), float(right["y1"]), float(right["x2"]), float(right["y2"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _make_event_row(
    event_type: str,
    frame: int,
    fps: float,
    team_id: str = "",
    team_name: str = "",
    player_track_id: str = "",
    secondary_track_id: str = "",
    confidence: float = 0.0,
    details: str = "",
) -> Dict[str, Any]:
    return {
        "row_type": "event",
        "event_id": "",
        "event_type": event_type,
        "event_label_ko": EVENT_LABELS_KO.get(event_type, event_type),
        "count": "",
        "frame": int(frame),
        "time_sec": round(float(frame) / float(fps), 3) if fps > 0 else "",
        "team_id": team_id,
        "team_name": team_name,
        "player_track_id": player_track_id,
        "secondary_track_id": secondary_track_id,
        "confidence": round(float(confidence), 3),
        "details": details,
    }


def _event_output_dataframe(
    events: Sequence[Mapping[str, Any]],
    fps: float,
    known_teams: Optional[Sequence[Mapping[str, str]]] = None,
    possession_stats: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    event_rows = sorted(
        [dict(row) for row in events],
        key=lambda row: (int(row.get("frame") or 0), EVENT_ORDER.index(row.get("event_type")) if row.get("event_type") in EVENT_ORDER else 999),
    )
    for index, row in enumerate(event_rows, start=1):
        row["event_id"] = f"E{index:05d}"

    wide_summary_rows = [
        _make_wide_summary_row(
            row_type="total_summary",
            event_type="all_totals",
            event_label_ko="전체종합",
            team_id="all",
            team_name="all",
            event_rows=event_rows,
            possession_stats=possession_stats,
        )
    ]
    for team in _team_summary_keys(event_rows, known_teams, possession_stats):
        team_id = str(team["team_id"])
        team_events = [
            row for row in event_rows if str(row.get("team_id", "")).strip() == team_id
        ]
        wide_summary_rows.append(
            _make_wide_summary_row(
                row_type="team_summary",
                event_type="team_totals",
                event_label_ko="팀별종합",
                team_id=team_id,
                team_name=str(team.get("team_name", "")),
                event_rows=team_events,
                possession_stats=possession_stats,
            )
        )

    total_summary_rows = []
    for event_type in EVENT_ORDER:
        total_summary_rows.append(
            {
                "row_type": "event_summary",
                "event_id": "",
                "event_type": event_type,
                "event_label_ko": EVENT_LABELS_KO[event_type],
                "count": sum(1 for row in event_rows if row["event_type"] == event_type),
                "frame": "",
                "time_sec": "",
                "team_id": "all",
                "team_name": "all",
                "player_track_id": "",
                "secondary_track_id": "",
                "confidence": "",
                "details": "total_count",
            }
        )

    team_summary_rows = []
    team_keys = sorted(
        {
            (str(row.get("team_id", "")), str(row.get("team_name", "")))
            for row in event_rows
            if str(row.get("team_id", "")).strip()
        }
    )
    for team_id, team_name in team_keys:
        for event_type in EVENT_ORDER:
            team_summary_rows.append(
                {
                    "row_type": "team_event_summary",
                    "event_id": "",
                    "event_type": event_type,
                    "event_label_ko": EVENT_LABELS_KO[event_type],
                    "count": sum(
                        1
                        for row in event_rows
                        if row["event_type"] == event_type and str(row.get("team_id", "")) == team_id
                    ),
                    "frame": "",
                    "time_sec": "",
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_track_id": "",
                    "secondary_track_id": "",
                    "confidence": "",
                    "details": "team_count",
                }
            )

    note_row = {
        "row_type": "note",
        "event_id": "",
        "event_type": "measurement_note",
        "event_label_ko": "측정메모",
        "count": "",
        "frame": "",
        "time_sec": "",
        "team_id": "",
        "team_name": "",
        "player_track_id": "",
        "secondary_track_id": "",
        "confidence": "",
        "details": "객체 박스와 공 궤적 기반 추정값입니다. 파울은 휘슬/오디오/포즈 정보가 없어 신뢰도가 낮습니다.",
    }

    rows = wide_summary_rows + total_summary_rows + team_summary_rows + [note_row] + event_rows
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _make_wide_summary_row(
    row_type: str,
    event_type: str,
    event_label_ko: str,
    team_id: str,
    team_name: str,
    event_rows: Sequence[Mapping[str, Any]],
    possession_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    counts = _event_count_values(event_rows)
    counts.update(_possession_feature_values(team_id, possession_stats))
    return {
        "row_type": row_type,
        "event_id": "",
        "event_type": event_type,
        "event_label_ko": event_label_ko,
        "count": sum(int(counts[f"{event_type}_count"]) for event_type in EVENT_ORDER),
        "frame": "",
        "time_sec": "",
        "team_id": team_id,
        "team_name": team_name,
        **counts,
        "player_track_id": "",
        "secondary_track_id": "",
        "confidence": "",
        "details": "wide_team_event_totals",
    }


def _event_count_values(event_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Any] = {
        f"{event_type}_count": sum(
            1 for row in event_rows if row.get("event_type") == event_type
        )
        for event_type in EVENT_ORDER
    }
    attempts = int(counts["shot_attempt_count"])
    made = int(counts["shot_made_count"])
    counts["shot_success_rate_pct"] = round((made / attempts) * 100.0, 2) if attempts > 0 else 0.0
    return counts


def _possession_feature_values(
    team_id: str,
    possession_stats: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not possession_stats:
        return {
            "possession_frame_count": 0,
            "possession_time_sec": 0.0,
            "possession_pct": 0.0,
        }

    if team_id == "all":
        return {
            "possession_frame_count": int(possession_stats.get("total_possession_frames", 0) or 0),
            "possession_time_sec": round(
                float(possession_stats.get("total_possession_time_sec", 0.0) or 0.0),
                3,
            ),
            "possession_pct": 100.0
            if int(possession_stats.get("total_possession_frames", 0) or 0) > 0
            else 0.0,
        }

    teams = possession_stats.get("teams", {})
    team_stats = teams.get(str(team_id), {}) if isinstance(teams, Mapping) else {}
    return {
        "possession_frame_count": int(team_stats.get("possession_frame_count", 0) or 0),
        "possession_time_sec": round(float(team_stats.get("possession_time_sec", 0.0) or 0.0), 3),
        "possession_pct": round(float(team_stats.get("possession_pct", 0.0) or 0.0), 2),
    }


def _team_summary_keys(
    event_rows: Sequence[Mapping[str, Any]],
    known_teams: Optional[Sequence[Mapping[str, str]]],
    possession_stats: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    teams: Dict[str, Dict[str, str]] = {}
    for team in known_teams or []:
        team_id = _clean_team_id(team.get("team_id"))
        if not team_id:
            continue
        teams[team_id] = {
            "team_id": team_id,
            "team_name": _clean_text(team.get("team_name")) or f"team_{team_id}",
        }

    for row in event_rows:
        team_id = _clean_team_id(row.get("team_id"))
        if not team_id:
            continue
        teams.setdefault(
            team_id,
            {
                "team_id": team_id,
                "team_name": _clean_text(row.get("team_name")) or f"team_{team_id}",
            },
        )

    if possession_stats:
        stat_teams = possession_stats.get("teams", {})
        if isinstance(stat_teams, Mapping):
            for team_id, team in stat_teams.items():
                clean_id = _clean_team_id(team_id)
                if not clean_id:
                    continue
                teams.setdefault(
                    clean_id,
                    {
                        "team_id": clean_id,
                        "team_name": _clean_text(team.get("team_name")) or f"team_{clean_id}"
                        if isinstance(team, Mapping)
                        else f"team_{clean_id}",
                    },
                )

    return [teams[key] for key in sorted(teams, key=_team_sort_key)]


def _known_teams_from_tracking(df: pd.DataFrame) -> List[Dict[str, str]]:
    if df.empty or "team_id" not in df.columns:
        return []

    teams: Dict[str, Dict[str, str]] = {}
    player_mask = (
        df["class"].astype(str).str.lower().isin(PLAYER_CLASSES)
        if "class" in df.columns
        else pd.Series(True, index=df.index)
    )
    for row in df[player_mask].to_dict("records"):
        team_id = _clean_team_id(row.get("team_id"))
        if not team_id:
            continue
        teams.setdefault(
            team_id,
            {
                "team_id": team_id,
                "team_name": _clean_text(row.get("team_name")) or f"team_{team_id}",
            },
        )
    return [teams[key] for key in sorted(teams, key=_team_sort_key)]


def _team_sort_key(team_id: str) -> Tuple[int, str]:
    try:
        return int(float(team_id)), team_id
    except (TypeError, ValueError):
        return 999999, str(team_id)


def _find_last_possession_before(
    segments: Sequence[PossessionSegment],
    frame: int,
    max_lookback_frames: int,
) -> Optional[PossessionSegment]:
    candidates = [
        segment
        for segment in segments
        if segment.end_frame <= frame and frame - segment.end_frame <= max_lookback_frames
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda segment: segment.end_frame)


def _find_first_possession_after(
    segments: Sequence[PossessionSegment],
    frame: int,
    max_lookahead_frames: int,
) -> Optional[PossessionSegment]:
    candidates = [
        segment
        for segment in segments
        if segment.start_frame >= frame and segment.start_frame - frame <= max_lookahead_frames
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda segment: segment.start_frame)


def _near_made_shot(frame: int, shots: Sequence[ShotWindow], fps: float) -> bool:
    margin = max(5, int(round(fps * 2.5)))
    return any(shot.made and shot.event_frame <= frame <= shot.end_frame + margin for shot in shots)


def _near_missed_shot(frame: int, shots: Sequence[ShotWindow], fps: float) -> bool:
    margin = max(5, int(round(fps * 2.0)))
    return any((not shot.made) and shot.start_frame - margin <= frame <= shot.end_frame + margin for shot in shots)


def _frame_in_any_shot_window(frame: int, shots: Sequence[ShotWindow], fps: float) -> bool:
    margin = max(4, int(round(fps * 0.7)))
    return any(shot.start_frame - margin <= frame <= shot.end_frame + margin for shot in shots)


def _point_to_record_bbox_distance(point: Tuple[float, float], record: Mapping[str, Any]) -> float:
    try:
        return _point_rect_distance(
            point[0],
            point[1],
            float(record["x1"]),
            float(record["y1"]),
            float(record["x2"]),
            float(record["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _point_rect_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    dx = max(left - px, 0.0, px - right)
    dy = max(top - py, 0.0, py - bottom)
    return math.hypot(dx, dy)


def _center(record: Optional[Mapping[str, Any]]) -> Optional[Tuple[float, float]]:
    if record is None:
        return None
    try:
        x = float(record["x_center"])
        y = float(record["y_center"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _safe_bbox(
    row: Mapping[str, Any],
    frame_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    width, height = frame_size
    try:
        x1 = int(max(0, min(width - 1, round(float(row["x1"])))))
        y1 = int(max(0, min(height - 1, round(float(row["y1"])))))
        x2 = int(max(0, min(width, round(float(row["x2"])))))
        y2 = int(max(0, min(height, round(float(row["y2"])))))
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clean_team_id(value: Any) -> str:
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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


__all__ = [
    "assign_fallback_player_track_ids",
    "assign_player_teams_from_video",
    "measure_basketball_events",
    "write_event_measurements",
]
