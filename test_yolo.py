from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO("yolov8n.pt")
img = cv2.imread("test.jpg")

results = model(img)

data = []

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names[cls]

    if name == "person":
        pass
    elif name in ["sports ball", "frisbee"]:
        name = "sports ball"
    else:
        continue

    x1, y1, x2, y2 = map(float, box.xyxy[0])

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    data.append({
        "class": name,
        "confidence": round(conf, 2),
        "x_center": round(x_center, 2),
        "y_center": round(y_center, 2)
    })

df = pd.DataFrame(data)
df.to_csv("detection_result.csv", index=False, encoding="utf-8-sig")

print(df)
print("detection_result.csv 저장 완료")