import cv2
from ultralytics import YOLO

# Step 1: Load the trained YOLOv8 model
model = YOLO('/home/guanran/robotic_flower/bee_detection/yolov8/runs/detect/yolov8n-honeybee/weights/best.pt')  # 替换为你的模型路径

# Step 2: Initialize video capture from the Logitech Brio (usually at index 0 or 1)
cap = cv2.VideoCapture(0)  # 0 是默认摄像头，如果有多个摄像头，可能是1或者其他索引

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Step 3: Process the video stream frame by frame
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Step 4: Use YOLO model to predict on the captured frame
    results = model.predict(frame, stream=True)  # stream=True for efficient video processing

    # Step 5: Render the results on the frame
    for result in results:
        frame = result.plot()  # 将预测结果绘制到帧上

    # Step 6: Display the resulting frame
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 7: Release the capture and close windows
cap.release()
cv2.destroyAllWindows()