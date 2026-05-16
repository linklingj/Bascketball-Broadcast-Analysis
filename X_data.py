<X_data.py>
import numpy as np
import pandas as pd
import torch


def run_preprocessing_from_team_csv():
    # 1. ball_possession.csv 로드
    INPUT_CSV = "ball_possession.csv"
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(
            f" 파일이 없습니다."
        )
        return

    # 프레임 순서대로 정렬, 인덱스 초기화
    df = df.sort_values(by="frame").reset_index(drop=True)

    # 영상 해상도 및 코트 규격 설정
    IMG_WIDTH, IMG_HEIGHT = 1920, 1080
    FPS = 30
    PITCH_WIDTH_M, PITCH_HEIGHT_M = 28, 15  # 농구 코트 정규 규격(미터)

    print(f"데이터 로드 완료: {len(df)} 개의 프레임 데이터")

    # 2. 데이터 정규화 (Scaling)
    df["ball_x_norm"] = df["ball_x"] / IMG_WIDTH
    df["ball_y_norm"] = df["ball_y"] / IMG_HEIGHT

    # 0~1 정규화
    # 화면 대각선 최대 길이를 기준
    max_pixel_dist = np.sqrt(IMG_WIDTH**2 + IMG_HEIGHT**2)
    df["dist_to_player_norm"] = df["distance"] / max_pixel_dist

    # 3. 유도 특징량 생성 (공의 실제 속력 계산)
    df["prev_ball_x"] = df["ball_x_norm"].shift(1)
    df["prev_ball_y"] = df["ball_y_norm"].shift(1)

    # 실제 미터 단위로 변환해 속력(m/s) 계산
    df["dx_m"] = (df["ball_x_norm"] - df["prev_ball_x"]) * PITCH_WIDTH_M
    df["dy_m"] = (df["ball_y_norm"] - df["prev_ball_y"]) * PITCH_HEIGHT_M
    df["ball_speed_ms"] = np.sqrt(df["dx_m"] ** 2 + df["dy_m"] ** 2) * FPS

    # 첫 프레임의 결측치는 0
    df["ball_speed_ms"] = df["ball_speed_ms"].fillna(0)

    # 4. 모델 입력 특징
    # [공의 X위치, 공의 Y위치, 공의 속력, 공-선수 거리, 현재 소유 중인 선수 ID]
    feature_columns = [
        "ball_x_norm",
        "ball_y_norm",
        "ball_speed_ms",
        "dist_to_player_norm",
        "owner_track_id",
    ]
    X_data = df[feature_columns].values

    # 5. 슬라이딩 윈도우 생성
    WINDOW_SIZE = 30  # 30프레임(1초)을 하나의 묶음으로 설정
    X_sequences = []

    for i in range(len(X_data) - WINDOW_SIZE + 1):
        X_sequences.append(X_data[i : i + WINDOW_SIZE])

    # 6. 파이토치 텐서 변환 및 파일 저장
    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)

    OUTPUT_TENSOR_PATH = "X_data.pt"
    torch.save(X_tensor, OUTPUT_TENSOR_PATH)

    print("\n=== 전처리 완료 ===")
 

if __name__ == "__main__":
    run_preprocessing_from_team_csv()
