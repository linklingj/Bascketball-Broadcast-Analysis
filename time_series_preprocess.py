# <time_series_preprocess.py>

import numpy as np
import pandas as pd
import torch

def run_preprocessing():
    # 1. 데이터 로드
    INPUT_CSV = "video_detection_postprocessed.csv"
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"INPUT 파일이 없습니다")
        return

    IMG_WIDTH, IMG_HEIGHT = 1920, 1080
    FPS = 30
    PITCH_WIDTH_M, PITCH_HEIGHT_M = 28, 15

    # 2. 정규화 및 속력 계산
    df["x_norm"] = df["x_center"] / IMG_WIDTH
    df["y_norm"] = df["y_center"] / IMG_HEIGHT
    df = df.sort_values(by=["track_id", "frame"])

    df["prev_x"] = df.groupby("track_id")["x_norm"].shift(1)
    df["prev_y"] = df.groupby("track_id")["y_norm"].shift(1)

    df["dx_m"] = (df["x_norm"] - df["prev_x"]) * PITCH_WIDTH_M
    df["dy_m"] = (df["y_norm"] - df["prev_y"]) * PITCH_HEIGHT_M
    df["real_dist_m"] = np.sqrt(df["dx_m"] ** 2 + df["dy_m"] ** 2).fillna(0)

    df["speed_ms"] = df["real_dist_m"] * FPS
    df.loc[df["speed_ms"] > 11.0, "speed_ms"] = 11.0

    # 3. 프레임별 특징량 추출
    frame_features = []
    total_frames = df["frame"].max() + 1

    for f in range(total_frames):
        f_data = df[df["frame"] == f]
        ball = f_data[f_data["class"] == "ball"]
        players = f_data[f_data["class"] == "player"]

        if ball.empty or players.empty:
            continue

        bx, by = ball["x_norm"].values[0], ball["y_norm"].values[0]
        b_speed = ball["speed_ms"].values[0]

        px = players["x_norm"].values * PITCH_WIDTH_M
        py = players["y_norm"].values * PITCH_HEIGHT_M
        bx_m, by_m = bx * PITCH_WIDTH_M, by * PITCH_HEIGHT_M

        distances = np.sqrt((px - bx_m) ** 2 + (py - by_m) ** 2)
        min_dist = np.min(distances)
        nearest_player_speed = players.iloc[np.argmin(distances)]["speed_ms"]

        frame_features.append([bx, by, b_speed, min_dist, nearest_player_speed])

    feature_df = pd.DataFrame(
        frame_features, columns=["ball_x", "ball_y", "ball_speed", "min_dist", "player_speed"]
    )

    # 4. 슬라이딩 윈도우 생성
    WINDOW_SIZE = 30
    X_data = feature_df.values

    X_sequences = []
    for i in range(len(X_data) - WINDOW_SIZE + 1):
        X_sequences.append(X_data[i : i + WINDOW_SIZE])

    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)

    # 5. 파이토치 텐서 파일로 저장
    OUTPUT_TENSOR_PATH = "X_data.pt"
    torch.save(X_tensor, OUTPUT_TENSOR_PATH)
    
    print(f"파일 저장 완료")
    
if __name__ == "__main__":
    run_preprocessing()
