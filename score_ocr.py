from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import easyocr
import pandas as pd


DEFAULT_VIDEO_PATH = "1_.mp4"
DEFAULT_OUTPUT_PATH = "score_timeline.csv"
DEFAULT_INTERVAL_SEC = 1.0
DEFAULT_CONFIRM_COUNT = 2
DEFAULT_RESYNC_COUNT = 3
MAX_RESYNC_SCORE_JUMP = 12
INITIAL_SCORE_WINDOW_SEC = 5.0

MIN_OCR_CONFIDENCE = 0.25
MIN_INFO_OCR_CONFIDENCE = 0.10
MIN_NUMBER_HEIGHT = 12
MAX_SCORE_VALUE = 200
UNKNOWN = "UNKNOWN"

KNOWN_TEAM_CODES = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "GS",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "NY",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "SA",
    "TOR",
    "UTA",
    "WAS",
}

TEAM_OCR_ALIASES = {
    "UTAH": "UTA",
    "UIA": "UTA",
    "UT4": "UTA",
    "DENN": "DEN",
    "OEN": "DEN",
    "NETS": "BKN",
    "METS": "BKN",
    "NCTS": "BKN",
    "BROOKLYN": "BKN",
    "MAGIC": "ORL",
    "ORLANDO": "ORL",
}


@dataclass
class OcrItem:
    raw: str
    conf: float
    x: float
    y: float
    width: float
    height: float
    team: str
    roi: str = ""


@dataclass
class NumberCandidate:
    value: int
    raw: str
    conf: float
    x: float
    y: float
    width: float
    height: float
    roi: str = ""


@dataclass
class ParsedScoreboard:
    team1_score: int
    team2_score: int
    team1_name: str
    team2_name: str
    quarter: str
    game_clock: str
    region_score: float

    @property
    def score(self) -> tuple[int, int]:
        return self.team1_score, self.team2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract basketball scoreboard OCR timeline.")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input mp4 video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--team1", default=UNKNOWN, help="Manual left team name, e.g. DEN")
    parser.add_argument("--team2", default=UNKNOWN, help="Manual right team name, e.g. UTA")
    parser.add_argument(
        "--left-score-box",
        default="",
        help="Manual left score crop as x1,y1,x2,y2 ratios, e.g. 0.715,0.79,0.775,0.865",
    )
    parser.add_argument(
        "--right-score-box",
        default="",
        help="Manual right score crop as x1,y1,x2,y2 ratios, e.g. 0.835,0.79,0.895,0.865",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC, help="Sampling interval in seconds")
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=DEFAULT_CONFIRM_COUNT,
        help="Number of repeated scores required for initial confirmation",
    )
    parser.add_argument(
        "--resync-count",
        type=int,
        default=DEFAULT_RESYNC_COUNT,
        help="Repeated non-decreasing invalid scores required to resync after missed samples",
    )
    parser.add_argument(
        "--record-mode",
        choices=["every-sample", "score-change"],
        default="score-change",
        help="CSV record mode",
    )
    parser.add_argument("--gpu", action="store_true", help="Use EasyOCR GPU mode")
    parser.add_argument("--debug", action="store_true", help="Print OCR number candidates")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug limit. 0 means full video; positive value limits OCR samples.",
    )
    return parser.parse_args()


def parse_box_arg(value: str) -> tuple[float, float, float, float] | None:
    if not value:
        return None

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Box must have 4 comma-separated values: {value}")

    x1, y1, x2, y2 = (float(part) for part in parts)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Box ratios must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1: {value}")

    return x1, y1, x2, y2


def bbox_bounds(bbox) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in bbox]
    ys = [float(point[1]) for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_quarter_text(text: str) -> str:
    compact = re.sub(r"[^0-9A-Z]", "", text.upper())
    compact = compact.replace("OQ", "0Q").replace("5T", "ST").replace("S1", "ST")

    suffix_only = {
        "ST": "1Q",
        "ND": "2Q",
        "RD": "3Q",
        "TH": "4Q",
    }
    for suffix, quarter in suffix_only.items():
        if compact == suffix or compact.endswith(suffix):
            return quarter

    match = re.fullmatch(r"([1-4])Q", compact)
    if match:
        return f"{match.group(1)}Q"

    if compact in {"OT", "0T"}:
        return "OT"

    ordinal_match = re.fullmatch(r"([1-4])(ST|ND|RD|TH)", compact)
    if ordinal_match:
        return f"{ordinal_match.group(1)}Q"

    return ""


def is_quarter_text(text: str) -> bool:
    return normalize_quarter_text(text) != ""


def normalize_game_clock_text(text: str) -> str:
    cleaned = text.upper().strip()
    cleaned = cleaned.replace("O", "0").replace("I", "1").replace("L", "1")
    cleaned = cleaned.replace("|", ":").replace(";", ":").replace(".", ":")

    match = re.search(r"(\d{1,2})\s*:\s*(\d{2})", cleaned)
    if not match:
        return ""

    minute = int(match.group(1))
    second = int(match.group(2))
    if 0 <= minute <= 15 and 0 <= second <= 59:
        return f"{minute:02d}:{second:02d}"
    return ""


def is_game_clock_text(text: str) -> bool:
    return normalize_game_clock_text(text) != ""


def game_clock_to_seconds(clock: str) -> int | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(clock).strip())
    if not match:
        return None

    minute = int(match.group(1))
    second = int(match.group(2))
    if 0 <= minute <= 15 and 0 <= second <= 59:
        return minute * 60 + second
    return None


def seconds_to_game_clock(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def validate_game_clock_transition(prev_clock: str, new_clock: str) -> tuple[bool, str]:
    prev_seconds = game_clock_to_seconds(prev_clock)
    new_seconds = game_clock_to_seconds(new_clock)
    if prev_seconds is None or new_seconds is None:
        return False, prev_clock

    if prev_seconds == 0 and new_seconds > 0:
        return True, new_clock

    diff = prev_seconds - new_seconds
    if 0 <= diff <= 3:
        return True, new_clock
    return False, prev_clock


def normalize_team_name(text: str) -> str:
    compact = text.upper().replace(" ", "")
    if normalize_quarter_text(compact) or is_game_clock_text(compact):
        return ""

    cleaned = re.sub(r"[^A-Z]", "", compact)
    if len(cleaned) < 2:
        return ""
    if cleaned in {"ST", "ND", "RD", "TH", "Q", "QTR", "PER", "PTS", "TOL", "BON"}:
        return ""

    for alias, team in sorted(TEAM_OCR_ALIASES.items(), key=lambda value: len(value[0]), reverse=True):
        if alias in cleaned:
            return team

    for team in KNOWN_TEAM_CODES:
        if team in cleaned:
            return team

    return ""


def get_score_number_regions(frame) -> dict[str, object]:
    h, w = frame.shape[:2]
    return {
        "bottom_left_score_only": frame[
            int(h * 0.84) : int(h * 0.98),
            int(w * 0.10) : int(w * 0.45),
        ],
        "bottom_scorebar_left": frame[
            int(h * 0.78) : int(h * 0.98),
            0 : int(w * 0.55),
        ],
        "bottom_full": frame[
            int(h * 0.62) : h,
            0:w,
        ],
    }


def get_scoreboard_info_regions(frame) -> dict[str, object]:
    h, w = frame.shape[:2]
    return {
        "bottom_info_right": frame[
            int(h * 0.78) : int(h * 0.98),
            int(w * 0.30) : int(w * 0.90),
        ],
        "bottom_info_full": frame[
            int(h * 0.72) : h,
            0:w,
        ],
        "top_info_full": frame[
            0 : int(h * 0.35),
            0:w,
        ],
    }


def crop_by_ratio(frame, y1: float, y2: float, x1: float, x2: float):
    h, w = frame.shape[:2]
    top = max(0, min(h - 1, int(h * y1)))
    bottom = max(top + 1, min(h, int(h * y2)))
    left = max(0, min(w - 1, int(w * x1)))
    right = max(left + 1, min(w, int(w * x2)))
    return frame[top:bottom, left:right]


def crop_by_box(frame, box: tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return crop_by_ratio(frame, y1, y2, x1, x2)


def get_score_box_pairs(frame) -> dict[str, tuple[object, object]]:
    return {
        "espn_bottom_left": (
            crop_by_ratio(frame, 0.84, 0.94, 0.180, 0.245),
            crop_by_ratio(frame, 0.84, 0.94, 0.245, 0.325),
        ),
        "espn_bottom_left_wide": (
            crop_by_ratio(frame, 0.82, 0.95, 0.160, 0.250),
            crop_by_ratio(frame, 0.82, 0.95, 0.235, 0.340),
        ),
        "espn_bottom_left_low": (
            crop_by_ratio(frame, 0.86, 0.98, 0.170, 0.250),
            crop_by_ratio(frame, 0.86, 0.98, 0.235, 0.340),
        ),
        "tnt_bottom_right": (
            crop_by_ratio(frame, 0.79, 0.865, 0.715, 0.775),
            crop_by_ratio(frame, 0.79, 0.865, 0.835, 0.895),
        ),
        "tnt_bottom_right_wide": (
            crop_by_ratio(frame, 0.76, 0.89, 0.690, 0.790),
            crop_by_ratio(frame, 0.76, 0.89, 0.810, 0.905),
        ),
        "bally_bottom_center": (
            crop_by_ratio(frame, 0.84, 0.965, 0.130, 0.200),
            crop_by_ratio(frame, 0.84, 0.965, 0.300, 0.370),
        ),
        "bally_bottom_center_wide": (
            crop_by_ratio(frame, 0.80, 0.98, 0.105, 0.220),
            crop_by_ratio(frame, 0.80, 0.98, 0.270, 0.390),
        ),
    }


def get_team_name_box_pairs(frame) -> dict[str, tuple[object, object]]:
    return {
        "espn_bottom_left_teams": (
            crop_by_ratio(frame, 0.84, 0.94, 0.000, 0.175),
            crop_by_ratio(frame, 0.84, 0.94, 0.325, 0.450),
        ),
        "espn_bottom_left_teams_wide": (
            crop_by_ratio(frame, 0.80, 0.96, 0.000, 0.180),
            crop_by_ratio(frame, 0.80, 0.96, 0.320, 0.500),
        ),
        "tnt_bottom_right_teams": (
            crop_by_ratio(frame, 0.79, 0.865, 0.655, 0.715),
            crop_by_ratio(frame, 0.79, 0.865, 0.775, 0.835),
        ),
        "tnt_bottom_right_teams_wide": (
            crop_by_ratio(frame, 0.76, 0.89, 0.630, 0.720),
            crop_by_ratio(frame, 0.76, 0.89, 0.765, 0.835),
        ),
        "bally_bottom_center_teams": (
            crop_by_ratio(frame, 0.76, 0.96, 0.030, 0.135),
            crop_by_ratio(frame, 0.76, 0.96, 0.200, 0.305),
        ),
        "bally_bottom_center_teams_wide": (
            crop_by_ratio(frame, 0.72, 0.98, 0.000, 0.155),
            crop_by_ratio(frame, 0.72, 0.98, 0.185, 0.330),
        ),
    }


def preprocess_for_ocr(region_img):
    h, w = region_img.shape[:2]
    scale = 1.5 if max(h, w) < 900 else 1.0
    image = region_img

    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.35, beta=8)
    return enhanced, scale


def read_ocr_items(
    reader: easyocr.Reader,
    region_img,
    min_confidence: float = MIN_OCR_CONFIDENCE,
    allowlist: str | None = None,
) -> list[OcrItem]:
    processed, scale = preprocess_for_ocr(region_img)
    read_kwargs = {"detail": 1, "paragraph": False}
    if allowlist:
        read_kwargs["allowlist"] = allowlist
    results = reader.readtext(processed, **read_kwargs)
    items: list[OcrItem] = []

    for bbox, text, conf in results:
        raw = str(text).strip()
        if conf < min_confidence or not raw:
            continue

        x1, y1, x2, y2 = bbox_bounds(bbox)
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        items.append(
            OcrItem(
                raw=raw,
                conf=float(conf),
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
                team=normalize_team_name(raw),
            )
        )

    return items


def read_score_number_items(reader: easyocr.Reader, region_img) -> list[OcrItem]:
    items = read_ocr_items(
        reader,
        region_img,
        min_confidence=MIN_INFO_OCR_CONFIDENCE,
        allowlist="0123456789",
    )
    for item in items:
        item.roi = "score_numbers"
    return items


def read_scoreboard_info_items(reader: easyocr.Reader, frame) -> list[OcrItem]:
    items: list[OcrItem] = []
    for region_name, region_img in get_scoreboard_info_regions(frame).items():
        region_items = read_ocr_items(
            reader,
            region_img,
            min_confidence=MIN_INFO_OCR_CONFIDENCE,
        )
        for item in region_items:
            item.roi = region_name
        items.extend(region_items)
    return items


def score_box_ocr_inputs(region_img, side: str) -> list[tuple[object, float, str]]:
    inputs: list[tuple[object, float, str]] = [(region_img, 1.0, "original")]

    if side == "right":
        enlarged = cv2.resize(region_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        enlarged = cv2.copyMakeBorder(
            enlarged,
            8,
            8,
            8,
            8,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        inputs.append((enlarged, 3.0, "right_enlarged"))

    return inputs


def read_single_score_value(reader: easyocr.Reader, region_img, side: str = "left") -> tuple[int, str, float] | None:
    candidates: list[NumberCandidate] = []

    for image, scale, source in score_box_ocr_inputs(region_img, side):
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="0123456789",
        )

        for bbox, text, conf in results:
            numeric = normalize_numeric_text(str(text))
            if not re.fullmatch(r"\d{1,3}", numeric):
                continue

            value = int(numeric)
            if value > MAX_SCORE_VALUE:
                continue

            x1, y1, x2, y2 = bbox_bounds(bbox)
            candidates.append(
                NumberCandidate(
                    value=value,
                    raw=str(text),
                    conf=float(conf),
                    x=((x1 + x2) / 2) / scale,
                    y=((y1 + y2) / 2) / scale,
                    width=(x2 - x1) / scale,
                    height=(y2 - y1) / scale,
                    roi=source,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item.conf + (0.08 if item.roi == "right_enlarged" else 0.0),
            item.height * item.conf,
        ),
        reverse=True,
    )
    best = candidates[0]
    return best.value, best.raw, best.conf


def read_single_team_name(reader: easyocr.Reader, region_img) -> tuple[str, str, float] | None:
    results = reader.readtext(
        region_img,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    candidates: list[tuple[float, str, str, float]] = []

    for bbox, text, conf in results:
        team = normalize_team_name(str(text))
        if not team:
            continue

        x1, y1, x2, y2 = bbox_bounds(bbox)
        width = x2 - x1
        height = y2 - y1
        score = float(conf) * 20.0 + min(width, 160) * 0.03 + height * 0.05
        candidates.append((score, team, str(text), float(conf)))

    if not candidates:
        return None

    candidates.sort(key=lambda value: value[0], reverse=True)
    _, team, raw, conf = candidates[0]
    return team, raw, conf


def extract_team_names_from_boxes(reader: easyocr.Reader, frame, debug: bool = False) -> tuple[str, str]:
    best_pair = ("", "")
    best_score = float("-inf")

    for region_name, (left_img, right_img) in get_team_name_box_pairs(frame).items():
        left_team = read_single_team_name(reader, left_img)
        right_team = read_single_team_name(reader, right_img)

        if debug:
            print(f"        {region_name}: left_team={left_team}, right_team={right_team}")

        score = 0.0
        if left_team:
            score += left_team[2] * 20.0 + 10.0
        if right_team:
            score += right_team[2] * 20.0 + 10.0
        if left_team and right_team:
            score += 15.0

        if score > best_score:
            best_score = score
            best_pair = (
                left_team[0] if left_team else "",
                right_team[0] if right_team else "",
            )

    return best_pair


def extract_team_names_from_info_items(items: list[OcrItem]) -> tuple[str, str]:
    team_items = [item for item in items if item.team]
    team_items.sort(key=lambda item: item.x)

    teams: list[str] = []
    for item in team_items:
        if item.team not in teams:
            teams.append(item.team)

    if len(teams) >= 2:
        return teams[0], teams[1]
    if len(teams) == 1:
        return teams[0], ""
    return "", ""


def normalize_numeric_text(text: str) -> str:
    cleaned = text.upper().strip()
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "O": "0",
                "D": "0",
                "I": "1",
                "L": "1",
                "Z": "2",
                "A": "4",
                "S": "5",
                "G": "6",
                "B": "8",
            }
        )
    )
    return re.sub(r"[^0-9]", "", cleaned)


def extract_number_candidates(items: list[OcrItem]) -> list[NumberCandidate]:
    candidates: list[NumberCandidate] = []

    for item in items:
        raw = item.raw
        compact = raw.upper().replace(" ", "")

        if item.team:
            continue
        if is_quarter_text(raw):
            continue
        if is_game_clock_text(raw):
            continue
        if re.fullmatch(r"[A-Z]+", compact):
            continue

        pair_match = re.fullmatch(r"\s*([0-9A-Z]{1,3})\s*[:|]\s*([0-9A-Z]{1,3})\s*", compact)
        if pair_match:
            left_text = normalize_numeric_text(pair_match.group(1))
            right_text = normalize_numeric_text(pair_match.group(2))
            if not re.fullmatch(r"\d{1,3}", left_text) or not re.fullmatch(r"\d{1,3}", right_text):
                continue

            left = int(left_text)
            right = int(right_text)
            if 0 <= left <= MAX_SCORE_VALUE and 0 <= right <= MAX_SCORE_VALUE:
                candidates.append(
                    NumberCandidate(
                        left,
                        raw,
                        item.conf,
                        item.x - item.width * 0.25,
                        item.y,
                        item.width / 2,
                        item.height,
                        item.roi,
                    )
                )
                candidates.append(
                    NumberCandidate(
                        right,
                        raw,
                        item.conf,
                        item.x + item.width * 0.25,
                        item.y,
                        item.width / 2,
                        item.height,
                        item.roi,
                    )
                )
            continue

        numeric = normalize_numeric_text(raw)
        if re.fullmatch(r"\d{4,6}", numeric):
            split_positions = [len(numeric) // 2]
            if len(numeric) == 5:
                split_positions = [2, 3]

            for split_at in split_positions:
                left_text = numeric[:split_at]
                right_text = numeric[split_at:]
                if not (1 <= len(left_text) <= 3 and 1 <= len(right_text) <= 3):
                    continue

                left = int(left_text)
                right = int(right_text)
                if 0 <= left <= MAX_SCORE_VALUE and 0 <= right <= MAX_SCORE_VALUE:
                    candidates.append(
                        NumberCandidate(
                            left,
                            raw,
                            item.conf,
                            item.x - item.width * 0.25,
                            item.y,
                            item.width / 2,
                            item.height,
                            item.roi,
                        )
                    )
                    candidates.append(
                        NumberCandidate(
                            right,
                            raw,
                            item.conf,
                            item.x + item.width * 0.25,
                            item.y,
                            item.width / 2,
                            item.height,
                            item.roi,
                        )
                    )
            continue

        if not re.fullmatch(r"\d{1,3}", numeric):
            continue

        value = int(numeric)
        if value > MAX_SCORE_VALUE:
            continue
        if item.height < MIN_NUMBER_HEIGHT:
            continue

        candidates.append(NumberCandidate(value, raw, item.conf, item.x, item.y, item.width, item.height, item.roi))

    return candidates


def deduplicate_numbers(candidates: list[NumberCandidate]) -> list[NumberCandidate]:
    sorted_candidates = sorted(candidates, key=lambda n: (n.height * n.conf, n.height), reverse=True)
    unique: list[NumberCandidate] = []

    for candidate in sorted_candidates:
        duplicated = False
        for saved in unique:
            same_value = candidate.value == saved.value
            close_x = abs(candidate.x - saved.x) <= max(candidate.width, saved.width)
            close_y = abs(candidate.y - saved.y) <= max(candidate.height, saved.height) * 0.6
            if same_value and close_x and close_y:
                duplicated = True
                break

        if not duplicated:
            unique.append(candidate)

    return unique


def is_obvious_bad_pair(left: NumberCandidate, right: NumberCandidate) -> bool:
    small = min(left.value, right.value)
    big = max(left.value, right.value)
    return small <= 4 and big >= 20


def choose_score_pair(candidates: list[NumberCandidate]) -> tuple[tuple[NumberCandidate, NumberCandidate] | None, float]:
    if len(candidates) < 2:
        return None, 0.0

    tallest = max(candidate.height for candidate in candidates)
    large_candidates = [
        candidate
        for candidate in candidates
        if candidate.height >= max(MIN_NUMBER_HEIGHT, tallest * 0.55)
    ]
    search_candidates = large_candidates if len(large_candidates) >= 2 else candidates

    best_pair = None
    best_score = float("-inf")

    score_roi_bonus = {
        "score_numbers": 45.0,
        "team1_score": 32.0,
        "team2_score": 32.0,
        "score_pair": 22.0,
    }
    info_rois = {"center_info", "middle_info", "right_info"}

    for i, first in enumerate(search_candidates):
        for second in search_candidates[i + 1 :]:
            left, right = sorted([first, second], key=lambda n: n.x)
            if is_obvious_bad_pair(left, right):
                continue

            x_gap = right.x - left.x
            y_gap = abs(right.y - left.y)
            max_height = max(left.height, right.height)
            min_height = min(left.height, right.height)
            height_ratio = min_height / max_height

            if x_gap < max_height * 0.7:
                continue
            if y_gap > max_height * 0.75:
                continue
            if height_ratio < 0.58:
                continue

            pair_score = 0.0
            pair_score += (left.height + right.height) * 2.4
            pair_score += (left.conf + right.conf) * 12
            pair_score += height_ratio * 25
            pair_score += score_roi_bonus.get(left.roi, 0.0)
            pair_score += score_roi_bonus.get(right.roi, 0.0)
            pair_score += 18.0 if {left.roi, right.roi} == {"team1_score", "team2_score"} else 0.0
            pair_score -= 22.0 if left.roi in info_rois else 0.0
            pair_score -= 22.0 if right.roi in info_rois else 0.0
            pair_score -= y_gap * 0.7
            pair_score -= x_gap * 0.025

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (left, right)

    if best_pair is None:
        return None, 0.0

    return best_pair, best_score


def extract_quarter(items: list[OcrItem]) -> str:
    quarter_candidates: list[tuple[float, str]] = []

    for item in items:
        quarter = normalize_quarter_text(item.raw)
        if quarter:
            score = item.conf * 20.0
            score += 20.0 if item.roi in {"center_info", "middle_info", "right_info"} else 0.0
            score += 8.0 if re.search(r"(?i)(st|nd|rd|th|q|ot)", item.raw) else 0.0
            score -= item.height * 0.03
            quarter_candidates.append((score, quarter))

    if not quarter_candidates:
        return ""

    quarter_candidates.sort(key=lambda value: value[0], reverse=True)
    return quarter_candidates[0][1]


def extract_game_clock(items: list[OcrItem]) -> str:
    clock_candidates: list[tuple[float, str]] = []

    for item in items:
        clock = normalize_game_clock_text(item.raw)
        clock_seconds = game_clock_to_seconds(clock) if clock else None
        if clock_seconds is None:
            continue

        score = 0.0
        score += item.conf * 20.0
        score += 25.0 if item.roi == "center_info" else 0.0
        score += 8.0 if item.height < 45 else 0.0
        score += 6.0 if re.search(r"[:|;.]", item.raw) else 0.0
        score -= item.height * 0.05
        clock_candidates.append((score, clock))

    if not clock_candidates:
        return ""

    clock_candidates.sort(key=lambda value: value[0], reverse=True)
    return clock_candidates[0][1]


def extract_scoreboard_info(reader: easyocr.Reader, frame, debug: bool = False) -> tuple[str, str, tuple[str, str]]:
    items = read_scoreboard_info_items(reader, frame)

    if debug:
        raw_items = ", ".join(f"{item.raw}(roi={item.roi},conf={item.conf:.2f})" for item in items)
        print(f"        info OCR: {raw_items}")

    return extract_quarter(items), extract_game_clock(items), extract_team_names_from_info_items(items)


def score_box_pair_quality(
    left_value: tuple[int, str, float],
    right_value: tuple[int, str, float],
    region_name: str,
) -> float:
    left_score, _left_raw, left_conf = left_value
    right_score, _right_raw, right_conf = right_value

    quality = (left_conf + right_conf) * 100.0
    quality += 30.0 if left_score >= 10 else -25.0
    quality += 30.0 if right_score >= 10 else -25.0
    quality += 10.0 if 0 <= left_score <= MAX_SCORE_VALUE else -100.0
    quality += 10.0 if 0 <= right_score <= MAX_SCORE_VALUE else -100.0

    if region_name.endswith("_wide"):
        quality -= 8.0
    if region_name.endswith("_low"):
        quality -= 12.0

    return quality


def parse_region(items: list[OcrItem]) -> tuple[ParsedScoreboard | None, list[NumberCandidate]]:
    numbers = deduplicate_numbers(extract_number_candidates(items))
    pair, pair_score = choose_score_pair(numbers)
    if pair is None:
        return None, numbers

    left, right = sorted(pair, key=lambda n: n.x)
    team1_name, team2_name = "", ""
    quarter = extract_quarter(items)
    game_clock = extract_game_clock(items)

    region_score = pair_score
    region_score += len(numbers) * 1.5
    region_score += 8 if game_clock else 0
    region_score += 6 if quarter else 0

    return (
        ParsedScoreboard(
            team1_score=left.value,
            team2_score=right.value,
            team1_name=team1_name,
            team2_name=team2_name,
            quarter=quarter,
            game_clock=game_clock,
            region_score=region_score,
        ),
        numbers,
    )


def choose_best_region(
    reader: easyocr.Reader,
    frame,
    debug: bool = False,
    manual_score_boxes: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
) -> tuple[str, ParsedScoreboard] | None:
    quarter, game_clock, info_teams = extract_scoreboard_info(reader, frame, debug=debug)
    box_team1, box_team2 = extract_team_names_from_boxes(reader, frame, debug=debug)
    info_team1, info_team2 = info_teams
    team1_name = box_team1 or info_team1
    team2_name = box_team2 or info_team2
    best_box_result: tuple[str, ParsedScoreboard] | None = None
    best_box_quality = float("-inf")

    score_box_pairs = {}
    if manual_score_boxes is not None:
        left_box, right_box = manual_score_boxes
        score_box_pairs["manual_score_boxes"] = (crop_by_box(frame, left_box), crop_by_box(frame, right_box))
    score_box_pairs.update(get_score_box_pairs(frame))

    for region_name, (left_img, right_img) in score_box_pairs.items():
        left_value = read_single_score_value(reader, left_img, side="left")
        right_value = read_single_score_value(reader, right_img, side="right")

        if debug:
            print(f"        {region_name}: left={left_value}, right={right_value}")

        if left_value is None or right_value is None:
            continue

        left_score, left_raw, left_conf = left_value
        right_score, right_raw, right_conf = right_value
        if is_obvious_bad_pair(
            NumberCandidate(left_score, left_raw, left_conf, 0, 0, 1, 1),
            NumberCandidate(right_score, right_raw, right_conf, 1, 0, 1, 1),
        ):
            continue

        quality = score_box_pair_quality(left_value, right_value, region_name)
        if region_name == "manual_score_boxes":
            quality += 50.0
        parsed = ParsedScoreboard(
            team1_score=left_score,
            team2_score=right_score,
            team1_name=team1_name,
            team2_name=team2_name,
            quarter=quarter,
            game_clock=game_clock,
            region_score=quality,
        )

        if quality > best_box_quality:
            best_box_quality = quality
            best_box_result = (region_name, parsed)

    if best_box_result is not None:
        return best_box_result

    for region_name, region_img in get_score_number_regions(frame).items():
        items = read_score_number_items(reader, region_img)
        parsed, numbers = parse_region(items)

        if debug:
            debug_numbers = ", ".join(
                f"{n.value}(raw={n.raw},roi={n.roi},h={n.height:.1f},x={n.x:.0f},y={n.y:.0f})"
                for n in numbers
            )
            print(f"        {region_name}: {debug_numbers}")

        if parsed is None:
            continue

        parsed.quarter = quarter
        parsed.game_clock = game_clock
        parsed.team1_name = team1_name
        parsed.team2_name = team2_name

        return region_name, parsed

    return None


def is_valid_score_transition(new_score: tuple[int, int], last_score: tuple[int, int] | None) -> bool:
    new_a, new_b = new_score

    if is_obvious_bad_pair(
        NumberCandidate(new_a, str(new_a), 1.0, 0, 0, 1, 1),
        NumberCandidate(new_b, str(new_b), 1.0, 1, 0, 1, 1),
    ):
        return False

    if last_score is None:
        return True

    last_a, last_b = last_score
    diff_a = new_a - last_a
    diff_b = new_b - last_b

    if diff_a < 0 or diff_b < 0:
        return False
    if diff_a == 0 and diff_b == 0:
        return True
    if diff_a in (1, 2, 3) and diff_b == 0:
        return True
    if diff_b in (1, 2, 3) and diff_a == 0:
        return True
    return False


def one_digit_off(value: int, target: int) -> bool:
    value_text = str(value)
    target_text = str(target)
    if len(value_text) != len(target_text):
        return False
    return sum(a != b for a, b in zip(value_text, target_text)) == 1


def plausible_single_digit_score_error(value: int, target: int) -> bool:
    return one_digit_off(value, target) and abs(value - target) <= 10


def infer_plausible_score_from_ocr_error(
    new_score: tuple[int, int],
    last_score: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if last_score is None or is_valid_score_transition(new_score, last_score):
        return None

    new_a, new_b = new_score
    last_a, last_b = last_score
    if new_b == last_b and new_a > last_a + 3:
        for target_a in (last_a + 1, last_a + 2, last_a + 3):
            if plausible_single_digit_score_error(new_a, target_a):
                return target_a, last_b

    if new_a == last_a and new_b > last_b + 3:
        for target_b in (last_b + 1, last_b + 2, last_b + 3):
            if plausible_single_digit_score_error(new_b, target_b):
                return last_a, target_b

    return None


def is_plausible_resync_score(new_score: tuple[int, int], last_score: tuple[int, int] | None) -> bool:
    if last_score is None or is_valid_score_transition(new_score, last_score):
        return False

    new_a, new_b = new_score
    last_a, last_b = last_score
    diff_a = new_a - last_a
    diff_b = new_b - last_b

    if diff_a < 0 or diff_b < 0:
        return False
    if diff_a > MAX_RESYNC_SCORE_JUMP or diff_b > MAX_RESYNC_SCORE_JUMP:
        return False
    if is_obvious_bad_pair(
        NumberCandidate(new_a, str(new_a), 1.0, 0, 0, 1, 1),
        NumberCandidate(new_b, str(new_b), 1.0, 1, 0, 1, 1),
    ):
        return False

    return diff_a > 0 or diff_b > 0


def with_score(parsed: ParsedScoreboard, score: tuple[int, int]) -> ParsedScoreboard:
    return ParsedScoreboard(
        team1_score=score[0],
        team2_score=score[1],
        team1_name=parsed.team1_name,
        team2_name=parsed.team2_name,
        quarter=parsed.quarter,
        game_clock=parsed.game_clock,
        region_score=parsed.region_score,
    )


def format_analysis_time(time_sec: float) -> str:
    total_seconds = max(0, int(round(time_sec)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def apply_stable_context(parsed: ParsedScoreboard, stable: dict[str, object], time_sec: float) -> ParsedScoreboard:
    last_team1 = str(stable.get("team1_name") or "")
    last_team2 = str(stable.get("team2_name") or "")
    if last_team1 == UNKNOWN:
        last_team1 = ""
    if last_team2 == UNKNOWN:
        last_team2 = ""

    if parsed.team1_name and not last_team1:
        stable["team1_name"] = parsed.team1_name
        last_team1 = parsed.team1_name
    if parsed.team2_name and not last_team2:
        stable["team2_name"] = parsed.team2_name
        last_team2 = parsed.team2_name

    team1 = last_team1 or parsed.team1_name or UNKNOWN
    team2 = last_team2 or parsed.team2_name or UNKNOWN
    quarter = parsed.quarter or str(stable.get("quarter") or UNKNOWN)
    game_clock = parsed.game_clock

    if parsed.quarter:
        stable["quarter"] = parsed.quarter

    clock_seconds = game_clock_to_seconds(game_clock) if game_clock else None
    last_clock_seconds = stable.get("game_clock_seconds")
    last_clock_time = stable.get("game_clock_time")
    last_quarter = stable.get("game_clock_quarter")
    last_clock = str(stable.get("game_clock") or "")

    if clock_seconds is not None:
        if last_clock and isinstance(last_clock_seconds, int) and last_quarter == quarter:
            elapsed = 1
            if isinstance(last_clock_time, (int, float)):
                elapsed = max(0, int(round(time_sec - float(last_clock_time))))

            expected_clock_seconds = max(0, last_clock_seconds - elapsed)
            drop = last_clock_seconds - clock_seconds

            if last_clock_seconds == 0 and clock_seconds > 0:
                game_clock = seconds_to_game_clock(clock_seconds)
            elif elapsed > 0 and clock_seconds >= last_clock_seconds:
                game_clock = seconds_to_game_clock(expected_clock_seconds)
                clock_seconds = expected_clock_seconds
            elif drop > max(3, elapsed + 2):
                game_clock = seconds_to_game_clock(expected_clock_seconds)
                clock_seconds = expected_clock_seconds
            else:
                game_clock = seconds_to_game_clock(clock_seconds)

            stable["game_clock"] = game_clock
            stable["game_clock_seconds"] = clock_seconds
            stable["game_clock_time"] = time_sec
            stable["game_clock_quarter"] = quarter
        else:
            game_clock = seconds_to_game_clock(clock_seconds)
            stable["game_clock"] = game_clock
            stable["game_clock_seconds"] = clock_seconds
            stable["game_clock_time"] = time_sec
            stable["game_clock_quarter"] = quarter
    elif isinstance(last_clock_seconds, int) and isinstance(last_clock_time, (int, float)) and last_quarter == quarter:
        elapsed = int(round(time_sec - float(last_clock_time)))
        clock_seconds = max(0, last_clock_seconds - elapsed)
        game_clock = seconds_to_game_clock(clock_seconds)
    else:
        game_clock = str(stable.get("game_clock") or UNKNOWN)

    return ParsedScoreboard(
        team1_score=parsed.team1_score,
        team2_score=parsed.team2_score,
        team1_name=team1,
        team2_name=team2,
        quarter=quarter,
        game_clock=game_clock,
        region_score=parsed.region_score,
    )


def append_record(records: list[dict[str, object]], time_sec: float, parsed: ParsedScoreboard) -> None:
    records.append(
        {
            "analysis_time": format_analysis_time(time_sec),
            "team1": parsed.team1_name,
            "team2": parsed.team2_name,
            "team1_score": parsed.team1_score,
            "team2_score": parsed.team2_score,
            "quarter": parsed.quarter,
            "game_clock": parsed.game_clock,
        }
    )


def print_scoreboard(time_sec: float, parsed: ParsedScoreboard) -> None:
    print(
        f"[{format_analysis_time(time_sec)}] "
        f"{parsed.team1_name} : {parsed.team2_name} | "
        f"{parsed.team1_score} : {parsed.team2_score} | "
        f"{parsed.quarter} | {parsed.game_clock}"
    )


def print_score_change_summary(df: pd.DataFrame) -> None:
    print("\n=== Score changes ===")

    if df.empty:
        print("No confirmed score changes.")
        print("Final score: UNKNOWN")
        return

    previous_score: tuple[int, int] | None = None
    final_row = df.iloc[0]
    shown_count = 0

    for _, row in df.iterrows():
        score = (int(row["team1_score"]), int(row["team2_score"]))
        if previous_score is None:
            previous_score = score
            final_row = row
            continue

        if score == previous_score:
            final_row = row
            continue

        if not is_valid_score_transition(score, previous_score):
            continue

        previous_score = score
        final_row = row
        shown_count += 1
        print(
            f"[{row['analysis_time']}] "
            f"{row['team1']} : {row['team2']} | "
            f"{row['team1_score']} : {row['team2_score']} | "
            f"{row['quarter']} | {row['game_clock']}"
        )

    if shown_count == 0:
        print("No confirmed score changes.")

    print(
        "\nFinal score: "
        f"{final_row['team1']} {final_row['team1_score']} : "
        f"{final_row['team2_score']} {final_row['team2']}"
    )


def analyze_video(
    reader: easyocr.Reader,
    video_path: str = DEFAULT_VIDEO_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    team1_name: str = UNKNOWN,
    team2_name: str = UNKNOWN,
    manual_score_boxes: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    confirm_count: int = DEFAULT_CONFIRM_COUNT,
    resync_count: int = DEFAULT_RESYNC_COUNT,
    record_mode: str = "score-change",
    debug: bool = False,
    max_samples: int = 0,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 30.0

    frame_step = max(1, int(round(fps * interval_sec)))

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, frames: {total_frames}, OCR every {frame_step} frames")

    records: list[dict[str, object]] = []
    stable: dict[str, object] = {
        "team1_name": "" if team1_name.strip().upper() == UNKNOWN else team1_name.strip().upper(),
        "team2_name": "" if team2_name.strip().upper() == UNKNOWN else team2_name.strip().upper(),
    }
    last_confirmed_score: tuple[int, int] | None = None
    initial_score_counts: Counter[tuple[int, int]] = Counter()
    initial_score_samples: dict[tuple[int, int], tuple[float, ParsedScoreboard, str]] = {}
    pending_corrected_score: tuple[int, int] | None = None
    pending_corrected_time: float = 0.0
    pending_corrected_parsed: ParsedScoreboard | None = None
    pending_corrected_votes = 0
    pending_resync_score: tuple[int, int] | None = None
    pending_resync_time: float = 0.0
    pending_resync_parsed: ParsedScoreboard | None = None
    pending_resync_votes = 0
    processed_samples = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_step != 0:
            frame_id += 1
            continue

        time_sec = frame_id / fps
        processed_samples += 1

        if max_samples and processed_samples > max_samples:
            break

        result = choose_best_region(
            reader,
            frame,
            debug=debug,
            manual_score_boxes=manual_score_boxes,
        )
        if result is None:
            print(f"[{format_analysis_time(time_sec)}] scoreboard not found")
            frame_id += 1
            continue

        region_name, parsed = result
        parsed = apply_stable_context(parsed, stable, time_sec)
        score = parsed.score

        print_scoreboard(time_sec, parsed)

        if last_confirmed_score is None:
            if time_sec < INITIAL_SCORE_WINDOW_SEC:
                initial_score_counts[score] += 1
                initial_score_samples[score] = (time_sec, parsed, region_name)
                print(
                    "           -> initial candidate collected: "
                    f"{score[0]}:{score[1]} ({initial_score_counts[score]} votes, not confirmed)"
                )
                frame_id += 1
                continue

            if initial_score_counts:
                initial_score, votes = initial_score_counts.most_common(1)[0]
                sample_time, sample_parsed, sample_region = initial_score_samples[initial_score]
                last_confirmed_score = initial_score
                append_record(records, sample_time, sample_parsed)
                print(
                    "           -> initial score confirmed by majority: "
                    f"{initial_score[0]}:{initial_score[1]} ({votes} votes, {sample_region})"
                )

                if score == last_confirmed_score:
                    frame_id += 1
                    continue
            else:
                last_confirmed_score = score
                append_record(records, time_sec, parsed)
                print(f"           -> initial score confirmed from {region_name}")
                frame_id += 1
                continue

        if not is_valid_score_transition(score, last_confirmed_score):
            corrected_score = infer_plausible_score_from_ocr_error(score, last_confirmed_score)
            if corrected_score is not None:
                corrected_parsed = with_score(parsed, corrected_score)
                if pending_corrected_score == corrected_score:
                    pending_corrected_votes += 1
                else:
                    pending_corrected_score = corrected_score
                    pending_corrected_time = time_sec
                    pending_corrected_parsed = corrected_parsed
                    pending_corrected_votes = 1

                print(
                    "           -> possible OCR score correction: "
                    f"{score[0]}:{score[1]} -> {corrected_score[0]}:{corrected_score[1]} "
                    f"({pending_corrected_votes} votes)"
                )

                if pending_corrected_votes >= max(1, confirm_count):
                    last_confirmed_score = corrected_score
                    append_record(records, pending_corrected_time, pending_corrected_parsed or corrected_parsed)
                    print(
                        "           -> corrected score change recorded: "
                        f"{corrected_score[0]}:{corrected_score[1]} "
                        f"at {format_analysis_time(pending_corrected_time)}"
                    )
                    pending_corrected_score = None
                    pending_corrected_parsed = None
                    pending_corrected_votes = 0
                    pending_resync_score = None
                    pending_resync_parsed = None
                    pending_resync_votes = 0

                frame_id += 1
                continue

            if is_plausible_resync_score(score, last_confirmed_score):
                if pending_resync_score == score:
                    pending_resync_votes += 1
                else:
                    pending_resync_score = score
                    pending_resync_time = time_sec
                    pending_resync_parsed = parsed
                    pending_resync_votes = 1

                print(
                    "           -> possible score resync: "
                    f"{last_confirmed_score[0]}:{last_confirmed_score[1]} -> {score[0]}:{score[1]} "
                    f"({pending_resync_votes} votes)"
                )

                if pending_resync_votes >= max(1, resync_count):
                    last_confirmed_score = score
                    append_record(records, pending_resync_time, pending_resync_parsed or parsed)
                    print(
                        "           -> score resynced after repeated read: "
                        f"{score[0]}:{score[1]} at {format_analysis_time(pending_resync_time)}"
                    )
                    pending_resync_score = None
                    pending_resync_parsed = None
                    pending_resync_votes = 0

                frame_id += 1
                continue

            pending_resync_score = None
            pending_resync_parsed = None
            pending_resync_votes = 0

            print(
                "           -> ignored invalid score transition: "
                f"{last_confirmed_score[0]}:{last_confirmed_score[1]} -> {score[0]}:{score[1]}"
            )
            frame_id += 1
            continue

        if score != last_confirmed_score:
            if pending_corrected_score == score and pending_corrected_parsed is not None:
                last_confirmed_score = score
                append_record(records, pending_corrected_time, pending_corrected_parsed)
                print(
                    "           -> score change recorded from earlier OCR correction: "
                    f"{score[0]}:{score[1]} at {format_analysis_time(pending_corrected_time)}"
                )
                pending_corrected_score = None
                pending_corrected_parsed = None
                pending_corrected_votes = 0
                pending_resync_score = None
                pending_resync_parsed = None
                pending_resync_votes = 0
                frame_id += 1
                continue

            last_confirmed_score = score
            append_record(records, time_sec, parsed)
            print(f"           -> score change recorded: {score[0]}:{score[1]}")
            pending_corrected_score = None
            pending_corrected_parsed = None
            pending_corrected_votes = 0
            pending_resync_score = None
            pending_resync_parsed = None
            pending_resync_votes = 0
        elif record_mode == "every-sample":
            append_record(records, time_sec, parsed)

        frame_id += 1

    cap.release()

    columns = [
        "analysis_time",
        "team1",
        "team2",
        "team1_score",
        "team2_score",
        "quarter",
        "game_clock",
    ]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
    print_score_change_summary(df)
    return df


def main() -> None:
    args = parse_args()
    left_score_box = parse_box_arg(args.left_score_box)
    right_score_box = parse_box_arg(args.right_score_box)
    manual_score_boxes = None
    if left_score_box or right_score_box:
        if left_score_box is None or right_score_box is None:
            raise ValueError("--left-score-box and --right-score-box must be used together")
        manual_score_boxes = (left_score_box, right_score_box)

    print("Preparing EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    analyze_video(
        reader=reader,
        video_path=str(Path(args.video)),
        output_path=str(Path(args.output)),
        team1_name=args.team1,
        team2_name=args.team2,
        manual_score_boxes=manual_score_boxes,
        interval_sec=args.interval,
        confirm_count=max(1, args.confirm_count),
        resync_count=max(1, args.resync_count),
        record_mode=args.record_mode,
        debug=args.debug,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
