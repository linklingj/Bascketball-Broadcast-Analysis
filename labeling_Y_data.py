import numpy as np
import pandas as pd
import torch

def generate_y_label_tensor():
    # 1. 데이터 로드 
    VIDEO_CSV = "video_detection.csv"
    EVENT_CSV = "event_summary.csv"
    
    try:
        v_df = pd.read_csv(VIDEO_CSV)
        e_df = pd.read_csv(EVENT_CSV)
    except FileNotFoundError as e:
        print(f"cannot find file: {e}")
        return

    # 영상의 실제 총 프레임 수 (0부터 시작->max + 1)
    total_frames = int(v_df["frame"].max() + 1)
    print(f"total frame: {total_frames}개")

    # 2. 이벤트 데이터 필터링 및 mapping 정의
    # row_type이 event인 것만 추출
    events_only = e_df[e_df["row_type"] == "event"]
    
    # 텍스트로 된 이벤트-> 정수(ID) 변환
    event_mapping = {
        "normal": 0,
        "pass": 1,
        "shot_made": 2,
        "shot_missed": 3,
        "rebound": 4,
        "steal_or_block": 5
    }
    # 3. 프레임 단위 정답 배열 생성
    # 우선 모든 프레임을 0으로 초기화
    frame_labels = np.zeros(total_frames, dtype=int)
    
    # 범위= 현재 영상 범위 이벤트
    current_events = events_only[(events_only["frame"] >= 0) & (events_only["frame"] < total_frames)]
    
    # 해당 프레임 위치에 정답 번호 대입
    for _, row in current_events.iterrows():
        f = int(row["frame"])
        etype = row["event_type"]
        if etype in event_mapping:
            frame_labels[f] = event_mapping[etype]
            print(f" event matching: [frame {f}] -> {row['event_label_ko']}(code {event_mapping[etype]})")

    # 4. 슬라이딩 윈도우 정답(y) 매핑
    WINDOW_SIZE = 30 
    y_list = []
    
    # 윈도우가 끝나는 시점 이벤트= 해당 윈도우의 정답(y) 설정
    for i in range(total_frames - WINDOW_SIZE + 1):
        window_last_frame = i + WINDOW_SIZE - 1
        y_list.append(frame_labels[window_last_frame])

    # 5. 4단계) 파이토치 텐서 변환 및 파일 저장
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    
    OUTPUT_FILE = "y_data.pt"
    torch.save(y_tensor, OUTPUT_FILE)
    
    print("post process_Y complete!")
    print(f"Tensor shape: {y_tensor.shape}")
    
if __name__ == "__main__":
    generate_y_label_tensor()
