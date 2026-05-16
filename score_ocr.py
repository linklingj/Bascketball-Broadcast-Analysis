import re
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import easyocr
import pandas as pd


# =========================
# Settings
# =========================
VIDEO_PATH = "test1.mp4"
OUTPUT_CSV = "score_timeline.csv"
INTERVAL_SEC = 1.0

# Increase this to 2 if the OCR is noisy and you want the same score to appear
# in consecutive sampled frames before it is recorded.
CONFIRM_COUNT = 2

MIN_OCR_CONFIDENCE = 0.25
MIN_SCORE_DIGIT_HEIGHT = 14
MAX_SCORE_VALUE = 200


reader = easyocr.Reader(["en"])


@dataclass(frozen=True)
class RegionCandidate:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frame):
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    @property
    def position(self) -> str:
        return f"{self.name}:{self.x1},{self.y1},{self.x2},{self.y2}"

    def expanded(self, frame_width: int, frame_height: int, ratio: float = 0.08) -> "RegionCandidate":
        pad_x = int((self.x2 - self.x1) * ratio)
        pad_y = int((self.y2 - self.y1) * ratio)
        return RegionCandidate(
            f"{self.name}_tracked",
            clamp(self.x1 - pad_x, 0, frame_width - 1),
            clamp(self.y1 - pad_y, 0, frame_height - 1),
            clamp(self.x2 + pad_x, self.x1 + 1, frame_width),
            clamp(self.y2 + pad_y, self.y1 + 1, frame_height),
        )


@dataclass
class NumberCandidate:
    value: int
    text: str
    conf: float
    x: float
    y: float
    width: float
    height: float

    @property
    def size_score(self) -> float:
        return self.height * max(self.conf, 0.1)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def make_region_candidates(frame_width: int, frame_height: int) -> list[RegionCandidate]:
    """Create broad scoreboard search windows around common broadcast overlay zones."""
    regions: list[RegionCandidate] = []

    def add(name: str, x1: float, y1: float, x2: float, y2: float) -> None:
        left = clamp(int(frame_width * x1), 0, frame_width - 1)
        top = clamp(int(frame_height * y1), 0, frame_height - 1)
        right = clamp(int(frame_width * x2), left + 1, frame_width)
        bottom = clamp(int(frame_height * y2), top + 1, frame_height)
        regions.append(RegionCandidate(name, left, top, right, bottom))

    # Compact corner/center bugs should be checked as their own regions so player
    # jersey numbers in the same broad band do not compete with the real score.
    for band_name, y1, y2 in (("top", 0.00, 0.24), ("bottom", 0.68, 1.00)):
        add(f"{band_name}_left_bug", 0.00, y1, 0.36, y2)
        add(f"{band_name}_center_bug", 0.32, y1, 0.68, y2)
        add(f"{band_name}_right_bug", 0.64, y1, 1.00, y2)

    # Sliding windows catch left/center/right overlays without assuming one broadcaster.
    for band_name, y1, y2 in (("top", 0.00, 0.34), ("bottom", 0.56, 1.00)):
        for idx, (x1, x2) in enumerate(
            (
                (0.00, 0.45),
                (0.15, 0.65),
                (0.32, 0.82),
                (0.55, 1.00),
            )
        ):
            add(f"{band_name}_window_{idx}", x1, y1, x2, y2)

    # Side panels are less common, but some streams put compact score bugs there.
    add("left_side", 0.00, 0.20, 0.33, 0.85)
    add("right_side", 0.67, 0.20, 1.00, 0.85)

    # Remove exact duplicate rectangles while preserving priority.
    unique: list[RegionCandidate] = []
    seen = set()
    for region in regions:
        key = (region.x1, region.y1, region.x2, region.y2)
        if key not in seen:
            unique.append(region)
            seen.add(key)
    return unique


def bbox_bounds(bbox: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    points = list(bbox)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def digits_from_text(text: str) -> Optional[str]:
    if re.search(r"[A-Za-z]", str(text)):
        return None
    cleaned = re.sub(r"\D", "", str(text))
    if not cleaned or len(cleaned) > 3:
        return None
    return cleaned


def read_region(region_image) -> list[tuple]:
    if region_image.size == 0:
        return []

    # Upscaling small overlays helps EasyOCR find compact score bugs.
    height, width = region_image.shape[:2]
    scale = 2 if max(height, width) < 900 else 1
    if scale > 1:
        region_image = cv2.resize(region_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.25, beta=8)

    results = reader.readtext(enhanced, detail=1, paragraph=False)
    if scale == 1:
        return results

    scaled_results = []
    for bbox, text, conf in results:
        scaled_bbox = [[point[0] / scale, point[1] / scale] for point in bbox]
        scaled_results.append((scaled_bbox, text, conf))
    return scaled_results


def collect_number_candidates(ocr_results: list[tuple]) -> list[NumberCandidate]:
    candidates: list[NumberCandidate] = []

    for bbox, text, conf in ocr_results:
        if conf < MIN_OCR_CONFIDENCE:
            continue
        if ":" in str(text):
            continue

        digits = digits_from_text(str(text))
        if digits is None:
            continue

        value = int(digits)
        if value > MAX_SCORE_VALUE:
            continue

        x1, y1, x2, y2 = bbox_bounds(bbox)
        box_height = y2 - y1
        if box_height < MIN_SCORE_DIGIT_HEIGHT:
            continue

        candidates.append(
            NumberCandidate(
                value=value,
                text=str(text),
                conf=float(conf),
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                width=x2 - x1,
                height=box_height,
            )
        )

    return candidates


def choose_score_pair(
    numbers: list[NumberCandidate],
    score_x_anchors: Optional[tuple[float, float]] = None,
    region_width: Optional[int] = None,
) -> Optional[tuple[NumberCandidate, NumberCandidate]]:
    if len(numbers) < 2:
        return None

    if score_x_anchors is not None:
        selected: list[NumberCandidate] = []
        for anchor in score_x_anchors:
            near_anchor = sorted(
                numbers,
                key=lambda item: (
                    abs(item.x - anchor) - item.size_score * 0.18,
                    -item.conf,
                ),
            )
            if not near_anchor:
                return None
            selected.append(near_anchor[0])

        if selected[0] is selected[1]:
            return None

        selected.sort(key=lambda item: item.x)
        return selected[0], selected[1]

    # Scores are usually a pair of large numbers on the same horizontal line.
    # Pair scoring prevents a far-away jersey/clock digit from beating the real
    # neighboring score numbers just because it has a tall OCR box.
    top_numbers = sorted(numbers, key=lambda item: item.size_score, reverse=True)[:8]
    best_pair: Optional[tuple[NumberCandidate, NumberCandidate]] = None
    best_pair_score = float("-inf")

    for index, first in enumerate(top_numbers):
        for second in top_numbers[index + 1 :]:
            x_gap = abs(first.x - second.x)
            y_gap = abs(first.y - second.y)
            max_height = max(first.height, second.height)
            min_height = min(first.height, second.height)

            if x_gap < 20:
                continue
            if y_gap > max_height * 0.75:
                continue

            height_similarity = min_height / max_height
            edge_penalty = 0.0
            gap_bonus = 0.0
            if region_width:
                # In the broad search windows, player jersey digits are often
                # near the crop edges. Real score digits tend to sit inside the
                # scoreboard panel with a moderate gap between the two scores.
                left_margin = min(first.x, second.x)
                right_margin = region_width - max(first.x, second.x)
                edge_margin = min(left_margin, right_margin)
                edge_penalty = max(0.0, region_width * 0.16 - edge_margin) * 0.35

                gap_ratio = x_gap / region_width
                gap_bonus = max(0.0, 1.0 - abs(gap_ratio - 0.28) / 0.16) * 18.0

            pair_score = (
                first.size_score
                + second.size_score
                + height_similarity * 12.0
                + gap_bonus
                - x_gap * 0.08
                - y_gap * 0.6
                - edge_penalty
            )

            if pair_score > best_pair_score:
                best_pair = (first, second)
                best_pair_score = pair_score

    if best_pair is None:
        return None

    selected_pair = sorted(best_pair, key=lambda item: item.x)
    return selected_pair[0], selected_pair[1]


def choose_score(
    numbers: list[NumberCandidate],
    score_x_anchors: Optional[tuple[float, float]] = None,
    region_width: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    pair = choose_score_pair(numbers, score_x_anchors, region_width)
    if pair is None:
        return None
    return pair[0].value, pair[1].value


def score_region(ocr_results: list[tuple], numbers: list[NumberCandidate]) -> float:
    text_count = sum(1 for _, text, conf in ocr_results if conf >= MIN_OCR_CONFIDENCE and str(text).strip())
    digit_count = sum(1 for _, text, conf in ocr_results if conf >= MIN_OCR_CONFIDENCE and re.search(r"\d", str(text)))
    colon_count = sum(1 for _, text, conf in ocr_results if conf >= MIN_OCR_CONFIDENCE and ":" in str(text))
    big_number_count = sum(1 for number in numbers if number.height >= MIN_SCORE_DIGIT_HEIGHT * 1.35)

    if digit_count == 0:
        return 0.0

    largest_numbers = sorted(numbers, key=lambda item: item.size_score, reverse=True)[:2]
    largest_score = sum(number.size_score for number in largest_numbers)

    return (
        digit_count * 6.0
        + colon_count * 4.0
        + text_count * 1.5
        + big_number_count * 5.0
        + largest_score * 0.15
    )


def evaluate_region(
    frame,
    region: RegionCandidate,
    score_x_anchors: Optional[tuple[float, float]] = None,
) -> tuple[Optional[tuple[int, int]], Optional[tuple[float, float]], float]:
    crop = region.crop(frame)
    ocr_results = read_region(crop)
    numbers = collect_number_candidates(ocr_results)
    score_pair = choose_score_pair(numbers, score_x_anchors, region.x2 - region.x1)
    raw_score = score_region(ocr_results, numbers)
    frame_height, frame_width = frame.shape[:2]
    area_ratio = ((region.x2 - region.x1) * (region.y2 - region.y1)) / float(frame_width * frame_height)
    compactness_penalty = area_ratio * 35.0
    adjusted_score = max(0.0, raw_score - compactness_penalty)

    if score_pair is None:
        return None, None, adjusted_score

    score_pair = tuple(sorted(score_pair, key=lambda item: item.x))
    detected_score = (score_pair[0].value, score_pair[1].value)
    detected_anchors = (score_pair[0].x, score_pair[1].x)
    return detected_score, detected_anchors, adjusted_score


def detect_scoreboard_and_score(
    frame,
    preferred_region: Optional[RegionCandidate] = None,
    score_x_anchors: Optional[tuple[float, float]] = None,
) -> tuple[Optional[tuple[int, int]], Optional[RegionCandidate], Optional[tuple[float, float]], float]:
    frame_height, frame_width = frame.shape[:2]
    best_score: Optional[tuple[int, int]] = None
    best_region: Optional[RegionCandidate] = None
    best_anchors: Optional[tuple[float, float]] = None
    best_region_score = 0.0

    if preferred_region is not None:
        score, anchors, region_score_value = evaluate_region(frame, preferred_region, score_x_anchors)
        if score is not None and region_score_value >= 18.0:
            return score, preferred_region, anchors, region_score_value

    for region in make_region_candidates(frame_width, frame_height):
        current_score, current_anchors, current_region_score = evaluate_region(frame, region)

        if current_score is not None and current_region_score > best_region_score:
            best_score = current_score
            best_region = region
            best_anchors = current_anchors
            best_region_score = current_region_score

    return best_score, best_region, best_anchors, best_region_score


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(fps * INTERVAL_SEC)))

    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Sample interval: {INTERVAL_SEC}s ({frame_step} frames)")

    score_records = []
    last_score: Optional[tuple[int, int]] = None
    pending_score: Optional[tuple[int, int]] = None
    pending_count = 0
    pending_frame_id: Optional[int] = None
    pending_time_sec: Optional[float] = None
    pending_position: Optional[str] = None
    last_region: Optional[RegionCandidate] = None
    score_x_anchors: Optional[tuple[float, float]] = None

    frame_id = 0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_id / fps
        score, region, anchors, region_score_value = detect_scoreboard_and_score(
            frame,
            last_region,
            score_x_anchors,
        )

        if score is None:
            print(f"{time_sec:.2f}s | scoreboard not found")
            frame_id += frame_step
            continue

        last_region = region
        score_x_anchors = anchors
        position = region.position if region is not None else None
        print(f"{time_sec:.2f}s | {position} | {score[0]} - {score[1]} | region_score={region_score_value:.1f}")

        if score == pending_score:
            pending_count += 1
        else:
            pending_score = score
            pending_count = 1
            pending_frame_id = frame_id
            pending_time_sec = time_sec
            pending_position = position

        if pending_count >= CONFIRM_COUNT and score != last_score:
            score_records.append(
                {
                    "frame": pending_frame_id if pending_frame_id is not None else frame_id,
                    "time_sec": round(pending_time_sec if pending_time_sec is not None else time_sec, 2),
                    "team1_score": score[0],
                    "team2_score": score[1],
                    "score_region_position": pending_position or position,
                }
            )
            print("score changed; recorded")
            last_score = score

        frame_id += frame_step

    cap.release()

    df = pd.DataFrame(
        score_records,
        columns=["frame", "time_sec", "team1_score", "team2_score", "score_region_position"],
    )
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
