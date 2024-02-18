from ultralytics import YOLO
import cv2

# Read the model
model = YOLO("runs/detect/train17/weights/best.pt")
# Capture the video
cap = cv2.VideoCapture(0)

while True:
    # Read the frames
    ret, frame = cap.read()

    # Read the results
    results = model.predict(frame, imgsz = 640)

    # Display the results
    annotations = results[0].plot()

    # Display the frames and the results video
    cv2.imshow("WASOR", annotations)

    # Close the program
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()