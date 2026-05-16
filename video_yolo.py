from ultralytics import YOLO
import cv2
import pandas as pd

TEST_MODE = True

# 모델 로드
model = YOLO("yolov8n.pt")

# 영상 불러오기
cap = cv2.VideoCapture("test1.mp4")

print("열림 여부:", cap.isOpened())

data = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # YOLO 실행
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25)
    
    boxes = results[0].boxes
    
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 클래스 이름
            name = model.names[cls]
            
            # 사람 + 공만 필터
            if name == "person":
                pass
            elif name in ["sports ball", "frisbee"]:
                name = "sports ball"
            else:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # 중심 좌표
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0])

            data.append({
                "frame": frame_id,
                "track_id": track_id,
                "class": name,
                "confidence": round(conf, 2),
                "x_center": round(x_center, 2),
                "y_center": round(y_center, 2)
            })
    
    frame_id += 1

    if TEST_MODE and frame_id >= 300:
        break

cap.release()

df = pd.DataFrame(data)

df = df.sort_values(by=["class", "frame"])

df["dx"] = df.groupby("class")["x_center"].diff()
df["dy"] = df.groupby("class")["y_center"].diff()

# CSV 저장
df.to_csv("video_detection.csv", index=False, encoding="utf-8-sig")

print("완료됨:", len(df), "개 데이터")