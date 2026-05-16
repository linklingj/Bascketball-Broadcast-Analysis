import pandas as pd
import math

df = pd.read_csv("video_detection.csv")

result = []

for frame in df["frame"].unique():
    frame_data = df[df["frame"] == frame]

    balls = frame_data[frame_data["class"] == "sports ball"]
    persons = frame_data[frame_data["class"] == "person"]

    if len(balls) == 0 or len(persons) == 0:
        continue

    ball = balls.iloc[0]
    ball_x = ball["x_center"]
    ball_y = ball["y_center"]

    min_distance = float("inf")
    owner_track_id = -1

    for _, person in persons.iterrows():
        px = person["x_center"]
        py = person["y_center"]

        distance = math.sqrt((ball_x - px) ** 2 + (ball_y - py) ** 2)

        if distance < min_distance:
            min_distance = distance
            owner_track_id = int(person["track_id"])

    result.append({
        "frame": frame,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "owner_track_id": owner_track_id,
        "distance": round(min_distance, 2)
    })

result_df = pd.DataFrame(result)
result_df.to_csv("ball_possession.csv", index=False, encoding="utf-8-sig")

print(result_df)
print("ball_possession.csv 저장 완료")

possession_count = result_df["owner_track_id"].value_counts()

print("\n공 점유 횟수:")
print(possession_count)

# 패스 횟수 계산
pass_count = 0
prev_owner = None

pass_events = []

for _, row in result_df.iterrows():
    current_owner = row["owner_track_id"]

    if prev_owner is not None and current_owner != prev_owner:
        pass_count += 1
        pass_events.append({
            "frame": row["frame"],
            "from": prev_owner,
            "to": current_owner
        })

    prev_owner = current_owner

pass_df = pd.DataFrame(pass_events)

print("\n패스 횟수:", pass_count)
print(pass_df)

# 전체 프레임 수
total_frames = len(result_df)

# 점유율 계산
possession_ratio = result_df["owner_track_id"].value_counts() / total_frames * 100

# 보기 좋게 정리
possession_ratio = possession_ratio.reset_index()
possession_ratio.columns = ["track_id", "possession_%"]

print("\n점유율 (%):")
print(possession_ratio)