import os

# Windows에서 PyTorch/NumPy/OpenMP 런타임 중복 로딩 회피 (다른 import 보다 먼저 설정)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional

import cv2
import pandas as pd
from PIL import Image as _PILImage

# Pillow 10+ 에서 Image.ANTIALIAS 가 제거됨. easyocr 1.7 이 이를 참조하므로 호환 셰임.
if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.LANCZOS

import easyocr


# =========================
# Settings
# =========================
VIDEO_PATH = "1_.mp4"
OUTPUT_CSV = "score_timeline.csv"
INTERVAL_SEC = 1.0

# 화면 하단 중앙 스코어보드의 두 점수 숫자 박스(고정 위치). (x1, y1, x2, y2)
# 1_.mp4 (2560x1440) 의 "DEN 46 | UTA 43" 점수 행에서 측정.
TEAM1_ROI = (1850, 1148, 2010, 1212)   # 좌측 팀(team_1, 예: DEN)
TEAM2_ROI = (2150, 1148, 2278, 1212)   # 우측 팀(team_2, 예: UTA)

# 같은 점수가 연속 N개 샘플에서 확인돼야 확정(순간적 OCR 깜빡임 제거)
CONFIRM_COUNT = 2

# 점수 숫자만 신뢰. 정답 숫자는 conf~1.0, 노이즈(BONUS 점선 등)는 conf<0.2 로 분리됨.
MIN_OCR_CONFIDENCE = 0.45
MAX_SCORE_VALUE = 200
# 한 번의 갱신에서 허용하는 최대 점수 증가폭(가림/리플레이 후 복귀 포함). 초과 시 오인식으로 간주.
MAX_SCORE_STEP = 9
# 작은 ROI 를 키워 OCR 정확도 향상
UPSCALE = 3

reader = easyocr.Reader(["en"])


def read_score(frame, roi: tuple[int, int, int, int]) -> tuple[Optional[int], float]:
    """고정 ROI 한 곳에서 점수 숫자 하나를 읽어 (값, 신뢰도)로 반환. 못 읽으면 (None, 0.0)."""
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=8)

    results = reader.readtext(gray, allowlist="0123456789", detail=1, paragraph=False)

    best_value: Optional[int] = None
    best_conf = 0.0
    best_area = 0.0
    for bbox, text, conf in results:
        if conf < MIN_OCR_CONFIDENCE:
            continue
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if not digits or len(digits) > 3:
            continue
        value = int(digits)
        if value > MAX_SCORE_VALUE:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        # 신뢰도 우선, 동률이면 더 큰(=실제 점수에 가까운) 박스 선택
        if (conf, area) > (best_conf, best_area):
            best_value, best_conf, best_area = value, float(conf), float(area)

    return best_value, best_conf


class TeamScoreTracker:
    """팀별 점수를 단조 증가 + 연속 확인 제약으로 추적."""

    def __init__(self) -> None:
        self.confirmed: Optional[int] = None
        self._cand: Optional[int] = None
        self._count = 0

    def update(self, observed: Optional[int]) -> bool:
        """관측값을 반영. 확정 점수가 바뀌면 True 반환."""
        if observed is None:
            return False

        # 단조성 + 과도한 점프 차단 (확정값이 있을 때만)
        if self.confirmed is not None:
            if observed < self.confirmed or observed > self.confirmed + MAX_SCORE_STEP:
                return False
            if observed == self.confirmed:
                self._cand, self._count = None, 0
                return False

        # 후보 누적
        if observed == self._cand:
            self._count += 1
        else:
            self._cand, self._count = observed, 1

        if self._count >= CONFIRM_COUNT:
            self.confirmed = self._cand
            self._cand, self._count = None, 0
            return True
        return False


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(fps * INTERVAL_SEC)))

    print(f"FPS: {fps:.2f} | Total frames: {total_frames} | "
          f"Sample: {INTERVAL_SEC}s ({frame_step} frames)")
    print(f"TEAM1_ROI={TEAM1_ROI} TEAM2_ROI={TEAM2_ROI}")

    t1 = TeamScoreTracker()
    t2 = TeamScoreTracker()
    records = []

    frame_id = 0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_id / fps
        v1, c1 = read_score(frame, TEAM1_ROI)
        v2, c2 = read_score(frame, TEAM2_ROI)
        changed = t1.update(v1)
        changed = t2.update(v2) or changed

        # 양 팀 점수가 모두 확정된 상태에서 변화가 생기면 기록
        if changed and t1.confirmed is not None and t2.confirmed is not None:
            records.append({
                "frame": frame_id,
                "time_sec": round(time_sec, 2),
                "team1_score": t1.confirmed,
                "team2_score": t2.confirmed,
                "team1_conf": round(c1, 2),
                "team2_conf": round(c2, 2),
            })
            print(f"{time_sec:7.1f}s | recorded {t1.confirmed} - {t2.confirmed} "
                  f"(conf {c1:.2f}/{c2:.2f})")
        elif frame_id % (frame_step * 60) == 0:  # 약 60초마다 진행 표시
            print(f"{time_sec:7.1f}s | obs {v1}({c1:.2f}) - {v2}({c2:.2f}) | "
                  f"confirmed {t1.confirmed} - {t2.confirmed}")

        frame_id += frame_step

    cap.release()

    df = pd.DataFrame(
        records,
        columns=["frame", "time_sec", "team1_score", "team2_score", "team1_conf", "team2_conf"],
    )
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_CSV}  ({len(df)} score changes)")
    if not df.empty:
        first, last = df.iloc[0], df.iloc[-1]
        print(f"First: {first['time_sec']:.0f}s  {first['team1_score']} - {first['team2_score']}")
        print(f"Last:  {last['time_sec']:.0f}s  {last['team1_score']} - {last['team2_score']}")


if __name__ == "__main__":
    main()
