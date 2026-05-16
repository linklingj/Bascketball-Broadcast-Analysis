import cv2

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)

    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    cv2.imshow("Motion", thresh)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()